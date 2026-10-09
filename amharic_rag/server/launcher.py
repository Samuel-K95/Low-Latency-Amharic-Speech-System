#!/usr/bin/env python3
"""
Server Launcher & Public Tunnel Orchestrator.
Supports 1-click startup on Kaggle & Colab with Cloudflare Tunnel or ngrok.
"""
import os
import re
import sys
import time
import shutil
import urllib.request
import subprocess
import uvicorn

def setup_cloudflared() -> str:
    """Ensures cloudflared binary exists and is executable."""
    cloudflared_path = shutil.which("cloudflared")
    if cloudflared_path:
        return cloudflared_path

    # On Linux / Colab / Kaggle, download the standalone binary
    target_bin = "/kaggle/working/cloudflared" if os.path.exists("/kaggle") else "/content/cloudflared"
    if not os.path.exists(target_bin):
        print("[Launcher] Downloading cloudflared tunnel client...")
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        try:
            urllib.request.urlretrieve(url, target_bin)
            os.chmod(target_bin, 0o755)
        except Exception as e:
            print(f"[Launcher] Failed to download cloudflared: {e}")
            return "cloudflared"

    return target_bin

def start_cloudflare_tunnel(port: int = 8000) -> str:
    """Starts cloudflared tunnel and returns the public HTTPS URL."""
    bin_path = setup_cloudflared()
    log_file = "/kaggle/working/tunnel.log" if os.path.exists("/kaggle") else "/content/tunnel.log"
    if not os.path.exists(os.path.dirname(log_file)):
        log_file = "tunnel.log"

    # Terminate any old tunnels
    subprocess.run(["pkill", "-9", "-f", "cloudflared"], check=False)
    time.sleep(1)

    print(f"[Launcher] Opening Cloudflare Tunnel to localhost:{port}...")
    log_out = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            bin_path,
            "tunnel",
            "--protocol", "http2",
            "--edge-ip-version", "4",
            "--url", f"http://127.0.0.1:{port}",
        ],
        stdout=log_out,
        stderr=subprocess.STDOUT
    )

    public_url = None
    t0 = time.time()
    while time.time() - t0 < 30:
        if os.path.exists(log_file):
            content = open(log_file, "r", encoding="utf-8", errors="ignore").read()
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
            if match:
                public_url = match.group(0)
                break
        time.sleep(1)

    if not public_url:
        print("[Launcher] Warning: Could not detect Cloudflare URL within 30s. Check tunnel.log.")
        public_url = f"http://localhost:{port}"

    return public_url

def start_ngrok_tunnel(port: int = 8000, auth_token: str = None) -> str:
    """Starts ngrok tunnel if preferred."""
    from pyngrok import ngrok
    if auth_token:
        ngrok.set_auth_token(auth_token)
    tunnel = ngrok.connect(port, "http")
    return tunnel.public_url

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    tunnel: str = "cloudflare",
    ngrok_token: str = None
):
    """
    Main entry point for starting the FastAPI backend + Tunnel on Kaggle / Colab.
    """
    public_url = None
    if tunnel == "cloudflare":
        public_url = start_cloudflare_tunnel(port=port)
    elif tunnel == "ngrok":
        public_url = start_ngrok_tunnel(port=port, auth_token=ngrok_token)
    else:
        public_url = f"http://{host}:{port}"

    ws_url = public_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws/s2s"

    print("\n" + "=" * 70)
    print("  🚀 AMHARIC SPEECH-TO-SPEECH RAG API IS LIVE!")
    print("=" * 70)
    print(f"  🌐 Interactive Web UI:  {public_url}")
    print(f"  📖 Swagger UI Docs:     {public_url}/docs")
    print(f"  🎙️ Real-time WebSocket: {ws_url}")
    print(f"  🏥 Health Check:        {public_url}/health")
    print("=" * 70 + "\n")

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    run_server(port=8000, tunnel="cloudflare")
