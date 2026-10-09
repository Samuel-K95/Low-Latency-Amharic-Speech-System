#!/usr/bin/env python3
"""
LLM Generation & llama-server Process Management Service.
Controls llama-server subprocess on dedicated GPU (Multi-GPU Kaggle isolation),
manages CUDA runtime libraries, and handles async SSE token streaming.
"""
import os
import re
import sys
import glob
import json
import time
import shutil
import atexit
import subprocess
from pathlib import Path
from typing import AsyncGenerator, Generator, Optional, List, Dict, Tuple

import requests
from huggingface_hub import HfApi, hf_hub_download

from server.config import (
    BIN_DIR,
    GGUF_DIR,
    WORKING_DIR,
    SERVER_LOG_PATH,
    SYSTEM_PROMPT_PATH,
    LLM_HOST,
    LLM_PORT,
    LLM_SERVER_URL,
    LLM_GPU_ID,
    LLAMA_REPOS,
    LLAMA_QUANT,
    LLAMA_QUANT_FALLBACKS,
    LLAMA_CTX,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_REPEAT_PENALTY,
    IDK_TEXT,
    IMPROVEMENTS,
    LLM_HEALTH_TTL_S,
)

STRONG_ENDINGS = ("።", "፤", "፧", "!", "?")
PHRASE_ENDINGS = ("፣", "።", "፤", "!", "?")
FIRST_PHRASE_MIN_WORDS = 2

def sanitize_to_ethiopic(text: str) -> str:
    """Drops thinking blocks and non-Ethiopic noise; preserves Ge'ez digits and punctuation."""
    text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.S)
    text = re.sub(r"[^ሀ-፿\s0-9\.\,\?\!\;፡።፣፤፧\(\)\-]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def _is_idk_buffer(text: str) -> bool:
    """True if the stream looks like the canned refusal (do not early-split it)."""
    t = text or ""
    if IDK_TEXT and IDK_TEXT[:12] in t:
        return True
    if "ይቅርታ" in t and ("መልስ" in t or "መረጃ" in t):
        return True
    if "መረጃ የለኝም" in t:
        return True
    return False


def split_ready(buf: str, first: bool = True):
    """
    Splits complete phrases off the front of the incoming token buffer.
    Returns: (phrases_list, remaining_buffer, first_still_pending)
    """
    phrases, start = [], 0
    ends = PHRASE_ENDINGS if first else STRONG_ENDINGS

    for i, ch in enumerate(buf):
        if ch == ".":
            is_end = i + 1 < len(buf) and buf[i + 1].isspace()
        else:
            is_end = ch in ends

        if not is_end:
            continue

        cand = buf[start:i + 1]
        need = FIRST_PHRASE_MIN_WORDS if first else 1
        if len(sanitize_to_ethiopic(cand).split()) >= need:
            phrases.append(cand)
            start = i + 1
            first = False

    # Early first phrase cut on word boundary if early_first_words is enabled.
    # Never early-cut the IDK refusal — it becomes two clips of the full phrase.
    n = IMPROVEMENTS.get("early_first_words", 0)
    rest = buf[start:]
    if n and first and not phrases and not _is_idk_buffer(rest):
        words = list(re.finditer(r"\S+\s+", rest))
        if len(words) >= n and len(sanitize_to_ethiopic(rest[:words[n - 1].end()]).split()) >= n:
            cut = words[n - 1].end()
            return [rest[:cut]], rest[cut:], False

    return phrases, rest, first

def _add_lib_dir(d: str):
    d = str(d)
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if d not in cur.split(":"):
        os.environ["LD_LIBRARY_PATH"] = d + (":" + cur if cur else "")

class LLMService:
    def __init__(
        self,
        bin_dir: Path = BIN_DIR,
        gguf_dir: Path = GGUF_DIR,
        port: int = LLM_PORT,
        gpu_id: int = LLM_GPU_ID,
    ):
        self.bin_dir = Path(bin_dir)
        self.gguf_dir = Path(gguf_dir)
        self.port = port
        self.gpu_id = gpu_id
        self.server_url = f"http://{LLM_HOST}:{self.port}"
        self.server_proc: Optional[subprocess.Popen] = None
        self.binary_path: Optional[Path] = None
        self.model_path: Optional[Path] = None
        self.system_prompt: str = ""
        self._last_health_ok: float = 0.0
        self._setup_cuda_environment()
        self._load_system_prompt()
        atexit.register(self.stop)

    def _setup_cuda_environment(self):
        """Configures Linux CUDA runtime library search paths."""
        for _d in ["/usr/local/cuda/lib64", "/usr/local/cuda/targets/x86_64-linux/lib", "/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu"]:
            if os.path.exists(_d):
                _add_lib_dir(_d)
        
        # Check cuda13_libs if on Kaggle / Linux
        cuda13_dir = WORKING_DIR / "cuda13_libs"
        for _d in glob.glob(str(cuda13_dir / "nvidia" / "*" / "lib")):
            _add_lib_dir(_d)

    def _load_system_prompt(self):
        """Loads system prompt from disk or uses default."""
        if SYSTEM_PROMPT_PATH.exists():
            with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
                self.system_prompt = f.read().strip()
        else:
            self.system_prompt = (
                "You are a natural Amharic AI voice assistant specialized in Ethiopia's Labour Proclamation.\n"
                "Answer using ONLY the CONTEXT. If not present, reply exactly: " + IDK_TEXT
            )

    def _missing_libs(self, path: Path) -> List[str]:
        """Checks for missing shared libraries via ldd."""
        if sys.platform != "linux":
            return []
        try:
            r = subprocess.run(["ldd", str(path)], capture_output=True, text=True, env=os.environ)
            return [ln.split("=>")[0].strip() for ln in r.stdout.splitlines() if "not found" in ln]
        except Exception:
            return []

    def _bin_works(self, path: Path) -> bool:
        """Verifies binary can execute --version without loader errors."""
        if not path.exists():
            return False
        try:
            os.chmod(str(path), 0o755)
            if self._missing_libs(path):
                return False
            r = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=20, env=os.environ)
            out = (r.stdout + r.stderr).lower()
            return r.returncode == 0 and "error while loading shared libraries" not in out
        except Exception:
            return False

    def _install_cuda13_libs(self):
        """Installs CUDA 13 runtime libraries if binary requires libcudart.so.13."""
        cuda13_dir = WORKING_DIR / "cuda13_libs"
        print(f"[LLMService] Installing CUDA 13 runtime libraries into {cuda13_dir}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(cuda13_dir), "nvidia-cuda-runtime", "nvidia-cublas"],
                check=True
            )
            for d in glob.glob(str(cuda13_dir / "nvidia" / "*" / "lib")):
                _add_lib_dir(d)
        except Exception as e:
            print(f"[LLMService] Warning: Could not auto-install CUDA 13 libs: {e}")

    def find_or_setup_binary(self) -> Path:
        """Finds or prepares working llama-server binary."""
        local_target = WORKING_DIR / "llama-server"

        if self._bin_works(local_target):
            self.binary_path = local_target
            return local_target

        candidates = [
            self.bin_dir / "llama-server-sm75",
            self.bin_dir / "llama-server",
            Path("/kaggle/working/bin/llama-server-sm75"),
            Path("/content/bin/llama-server-sm75"),
        ] + [Path(p) for p in glob.glob("/kaggle/input/**/llama-server*", recursive=True)]

        for cand in candidates:
            if cand.is_file() and cand.stat().st_size > 10_000_000:
                print(f"[LLMService] Found candidate binary: {cand}")
                try:
                    if local_target.exists():
                        local_target.unlink()
                except Exception:
                    pass
                shutil.copy(str(cand), str(local_target))
                os.chmod(str(local_target), 0o755)

                miss = self._missing_libs(local_target)
                if miss:
                    print(f"[LLMService] Missing shared libraries: {miss}")
                    if any(m.endswith(".13") for m in miss):
                        self._install_cuda13_libs()

                if self._bin_works(local_target):
                    print(f"[LLMService] Verified working llama-server binary at {local_target}")
                    self.binary_path = local_target
                    return local_target

        if shutil.which("llama-server"):
            system_bin = Path(shutil.which("llama-server"))
            if self._bin_works(system_bin):
                self.binary_path = system_bin
                return system_bin

        raise FileNotFoundError(
            f"No working llama-server binary found. Checked candidates: {candidates}. "
            "Please ensure bin/llama-server-sm75 exists."
        )

    def find_or_download_gguf(self) -> Path:
        """Finds existing GGUF or downloads Gemma 4B GGUF from HuggingFace."""
        # 1. Check existing GGUFs in gguf_dir
        for f in self.gguf_dir.glob("*.gguf"):
            if f.stat().st_size > 1_000_000_000:
                self.model_path = f
                print(f"[LLMService] Found existing GGUF model: {f} ({f.stat().st_size / 1e9:.2f} GB)")
                return f

        # Also check /kaggle/working/gguf or dataset
        for cand_dir in [Path("/kaggle/working/gguf"), Path("/kaggle/input")]:
            if cand_dir.exists():
                for f in cand_dir.glob("**/*.gguf"):
                    if f.stat().st_size > 1_000_000_000:
                        self.model_path = f
                        print(f"[LLMService] Found GGUF model: {f} ({f.stat().st_size / 1e9:.2f} GB)")
                        return f

        # 2. Download from Hugging Face
        api = HfApi()
        hf_token = os.environ.get("HF_TOKEN") or None

        def _list_files(repo_id):
            tokens = [hf_token, None] if hf_token else [None]
            last_err = None
            for tok in tokens:
                try:
                    return api.list_repo_files(repo_id, token=tok)
                except Exception as e:
                    last_err = e
            return []

        choice = None
        target_quant = None
        for q in [LLAMA_QUANT] + LLAMA_QUANT_FALLBACKS:
            for repo in LLAMA_REPOS:
                files = _list_files(repo)
                hits = [
                    f for f in files
                    if f.endswith(".gguf") and "mmproj" not in f.lower()
                    and q.lower() in f.lower().replace("-", "_")
                ]
                if hits:
                    choice = (repo, sorted(hits, key=len)[0])
                    target_quant = q
                    break
            if choice:
                break

        if not choice:
            raise FileNotFoundError(
                f"No matching GGUF found in repos {LLAMA_REPOS} for quants {[LLAMA_QUANT] + LLAMA_QUANT_FALLBACKS}."
            )

        repo, fname = choice
        print(f"[LLMService] Downloading {fname} ({target_quant}) from {repo} to {self.gguf_dir}...")
        try:
            dl_path = hf_hub_download(repo_id=repo, filename=fname, local_dir=str(self.gguf_dir), token=hf_token)
        except Exception as e:
            if hf_token and "401" in str(e):
                print("[LLMService] HF token rejected; downloading anonymously...")
                dl_path = hf_hub_download(repo_id=repo, filename=fname, local_dir=str(self.gguf_dir), token=None)
            else:
                raise e

        self.model_path = Path(dl_path)
        print(f"[LLMService] GGUF model ready at {self.model_path} ({self.model_path.stat().st_size / 1e9:.2f} GB)")
        return self.model_path

    def _supports_flag(self, flag: str) -> bool:
        if not self.binary_path:
            return False
        try:
            res = subprocess.run([str(self.binary_path), "-h"], capture_output=True, text=True, timeout=30, env=os.environ)
            return flag in (res.stdout + res.stderr)
        except Exception:
            return flag == "--jinja"

    def start(self, timeout: int = 120):
        """Starts llama-server subprocess on the configured GPU and waits for /health 200."""
        self.stop()
        
        # Kill any lingering llama-server instances
        subprocess.run(["pkill", "-9", "-f", "llama-server"], check=False)
        time.sleep(1)

        self.find_or_setup_binary()
        self.find_or_download_gguf()

        env = os.environ.copy()
        if self.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        cmd = [
            str(self.binary_path),
            "-m", str(self.model_path),
            "-ngl", "99",
            "-c", str(LLAMA_CTX),
            "--host", LLM_HOST,
            "--port", str(self.port),
            "-np", "1",
            "--log-disable",
        ]
        if self._supports_flag("--jinja"):
            cmd.append("--jinja")

        print(f"[LLMService] Starting llama-server on port {self.port} (GPU {self.gpu_id})...")
        log_file = open(SERVER_LOG_PATH, "w", encoding="utf-8")
        self.server_proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env
        )

        # Wait for health check
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.server_proc.poll() is not None:
                log_file.flush()
                log_details = open(SERVER_LOG_PATH, "r", encoding="utf-8", errors="replace").read()
                print(f"\n[LLMService Error] llama-server process exited with code {self.server_proc.returncode}!")
                print(f"Executed command: {' '.join(cmd)}")
                print(f"Log output:\n{log_details[-3000:] if log_details.strip() else '(empty log)'}\n")
                raise RuntimeError(f"llama-server exited with code {self.server_proc.returncode}")

            try:
                r = requests.get(f"{self.server_url}/health", timeout=1)
                if r.status_code == 200:
                    self._last_health_ok = time.time()
                    print(f"[LLMService] llama-server ready on port {self.port} ({time.time() - t0:.1f}s).")
                    return
            except requests.RequestException:
                pass
            time.sleep(1)

        log_file.flush()
        log_details = open(SERVER_LOG_PATH, "r", encoding="utf-8", errors="replace").read()
        print(f"\n[LLMService Timeout] llama-server did not respond on /health within {timeout}s!")
        print(f"Log output:\n{log_details[-3000:]}\n")
        raise TimeoutError(f"llama-server failed to respond on port {self.port} within {timeout}s.")

    def stop(self):
        """Terminates running llama-server process."""
        if self.server_proc is not None and self.server_proc.poll() is None:
            print("[LLMService] Stopping llama-server...")
            self.server_proc.terminate()
            try:
                self.server_proc.wait(timeout=5)
            except Exception:
                self.server_proc.kill()
        self.server_proc = None

    def ensure_server(self, force_check: bool = False):
        """
        Auto-recovers llama-server if unresponsive.

        Skips a full HTTP health round-trip when the subprocess is alive and a
        recent health check succeeded (Phase 5: avoid redundant readiness I/O).
        """
        now = time.time()
        proc_alive = self.server_proc is not None and self.server_proc.poll() is None
        last_ok = getattr(self, "_last_health_ok", 0.0)
        if (
            not force_check
            and proc_alive
            and (now - last_ok) < LLM_HEALTH_TTL_S
        ):
            return

        try:
            if requests.get(f"{self.server_url}/health", timeout=0.5).status_code == 200:
                self._last_health_ok = now
                return
        except requests.RequestException:
            pass
        print("[LLMService] Server unhealthy or unresponsive. Restarting...")
        self.start()
        self._last_health_ok = time.time()

    def build_prompt_user_content(self, question: str, contexts: List[Dict]) -> str:
        """Formats RAG user prompt with retrieved context articles."""
        context_parts = []
        for i, c in enumerate(contexts, 1):
            art = c.get("article_id") or c.get("id", f"Passage {i}")
            txt = c.get("text_spoken") or c.get("text_clean", "")
            context_parts.append(f"[{art}]\n{txt}")
        
        context_block = "\n\n".join(context_parts)
        return f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"

    def generate_stream(
        self,
        user_text: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = LLM_MAX_TOKENS,
        history_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[Dict, None, None]:
        """
        Streams generated tokens via SSE and yields sentence/phrase chunks in real-time.

        ``history_messages`` is optional prior user/assistant turns only.
        Retrieval context stays inside ``user_text`` for the *current* question.
        """
        self.ensure_server()
        sys_prompt = system_prompt or self.system_prompt
        if not sys_prompt:
            self._load_system_prompt()
            sys_prompt = system_prompt or self.system_prompt

        messages: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]
        if history_messages:
            for m in history_messages:
                role = m.get("role")
                content = (m.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": LLM_TEMPERATURE,
            "repeat_penalty": LLM_REPEAT_PENALTY,
            "stream": True,
            "stream_options": {"include_usage": True},
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        t_start = time.perf_counter()
        resp = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=120
        )
        resp.raise_for_status()

        buf = ""
        first_phrase_pending = True
        ttft_s = 0.0
        first_sentence_s = 0.0
        full_tokens = []
        usage = {}

        for line in resp.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:].strip()
            if data_str == "[DONE]":
                break

            chunk = json.loads(data_str)
            if "usage" in chunk and chunk["usage"]:
                usage = chunk["usage"]

            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if not content:
                continue

            if ttft_s == 0.0:
                ttft_s = time.perf_counter() - t_start
                yield {"type": "first_token", "t": ttft_s}

            buf += content
            full_tokens.append(content)

            phrases, buf, first_phrase_pending = split_ready(buf, first_phrase_pending)
            for phrase in phrases:
                cleaned_phrase = sanitize_to_ethiopic(phrase)
                if cleaned_phrase:
                    now = time.perf_counter() - t_start
                    if first_sentence_s == 0.0:
                        first_sentence_s = now
                    yield {"type": "sentence", "text": cleaned_phrase, "t": now}

        # Flush any remainder in buffer
        tail = sanitize_to_ethiopic(buf)
        if tail:
            now = time.perf_counter() - t_start
            if first_sentence_s == 0.0:
                first_sentence_s = now
            yield {"type": "sentence", "text": tail, "t": now}

        t_total = time.perf_counter() - t_start
        full_response = sanitize_to_ethiopic("".join(full_tokens))
        new_tokens = usage.get("completion_tokens", len(full_tokens))
        prompt_tokens = usage.get("prompt_tokens", 0)

        yield {
            "type": "done",
            "text": full_response,
            "total_s": t_total,
            "ttft_s": ttft_s,
            "first_sentence_s": first_sentence_s,
            "new_tokens": new_tokens,
            "prompt_tokens": prompt_tokens,
            "tok_s": new_tokens / t_total if t_total > 0 else 0.0,
            "prefill_s": ttft_s,
            "server_s": t_total,
        }
