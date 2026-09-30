#!/usr/bin/env bash
# Talk to the Amharic speech-to-speech demo (laptop CPU). Ctrl+C to quit.
#   ./talk.sh                 speak into the microphone (fast Llama-1B-Amharic)
#   ./talk.sh --llm gemma     smarter answers (Gemma 4 E2B), about 2 s slower to start speaking
#   ./talk.sh --text          type Amharic instead of speaking
#   ./talk.sh --team          the final pipeline's models: hohe ASR + Tiny Aya + OmniVoice (slow on this CPU; also ./talk_team.sh)
#   ./talk.sh --team --tts mms   the same ASR + LLM with MMS-TTS, the TTS of the final pipeline (much faster)
#   ./talk.sh --asr hohe      mix and match: --asr ctc|hohe|whisper-small|whisper-tiny, --llm llama1b|gemma|tinyaya, --tts mms|omnivoice
#   ./talk.sh --help          all options
HERE="$(cd "$(dirname "$0")" && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore
PY="$HERE/.venv/bin/python"; [ -x "$PY" ] || PY=python3     # the venv from the README, else the system python
exec "$PY" "$HERE/smoke/talk.py" "$@"     # talk.py resolves file arguments, then moves into smoke/
