#!/usr/bin/env bash
# Talk to the demo using the final pipeline's models: hohe-asr-amharic + Tiny Aya Earth + OmniVoice (8 steps). Ctrl+C to quit.
#   ./talk_team.sh                  speak into the microphone
#   ./talk_team.sh --text           type Amharic instead of speaking
#   ./talk_team.sh --tts mms        swap OmniVoice for MMS-TTS (the TTS of the final pipeline, much faster)
#   ./talk_team.sh --omni-steps 16  clearer OmniVoice, slower
# Everything else is the same as ./talk.sh (see ./talk.sh --help).
exec "$(cd "$(dirname "$0")" && pwd)/talk.sh" --team "$@"
