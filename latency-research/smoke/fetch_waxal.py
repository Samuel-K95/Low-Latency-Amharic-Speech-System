"""Fetch a small, speaker-diverse sample of the WAXAL Amharic ASR test split.

Uses the HF datasets-server rows API so we never download the 900 MB parquet
shards. Each clip is converted to 16 kHz mono PCM WAV for the benchmarks.
"""
import json, subprocess, urllib.request, pathlib

OUT = pathlib.Path(__file__).parent / "audio" / "waxal_amh_test"
OUT.mkdir(parents=True, exist_ok=True)
API = "https://datasets-server.huggingface.co/rows?dataset=google/WaxalNLP&config=amh_asr&split=test&offset={o}&length={n}"

manifest, seen_speakers = [], {}
for offset in range(0, 3400, 100):          # 34 windows across the split
    rows = json.load(urllib.request.urlopen(API.format(o=offset, n=10)))["rows"]
    for r in rows:
        row = r["row"]
        spk = row["speaker_id"]
        if seen_speakers.get(spk, 0) >= 2:   # at most 2 clips per speaker
            continue
        seen_speakers[spk] = seen_speakers.get(spk, 0) + 1
        mp3 = OUT / f"{r['row_idx']:05d}.mp3"
        wav = mp3.with_suffix(".wav")
        urllib.request.urlretrieve(row["audio"][0]["src"], mp3)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3),
                        "-ac", "1", "-ar", "16000", str(wav)], check=True)
        mp3.unlink()
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
              "format=duration", "-of", "csv=p=0", str(wav)]))
        manifest.append({"file": wav.name, "row_idx": r["row_idx"], "speaker_id": spk,
                         "gender": row["gender"], "duration_s": round(dur, 2),
                         "text": row["transcription"]})
        break                                # one clip per window
    if len(manifest) >= 30:
        break

(OUT / "manifest.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n")
tot = sum(m["duration_s"] for m in manifest)
print(f"{len(manifest)} clips, {len(seen_speakers)} speakers, {tot:.0f}s total, "
      f"mean {tot/len(manifest):.1f}s, genders={sorted(set(m['gender'] for m in manifest))}")
