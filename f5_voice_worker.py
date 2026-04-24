"""F5-TTS voice worker. Runs in isolated venv via subprocess.

Root cause of 'кхм'/silent output (fix 2026-04-23):
F5-TTS requires a CORRECT ref_text (transcript of reference audio).
When ref_text="" it tries to auto-transcribe via whisper-large-v3-turbo.
On our podcast reference (noisy intro) whisper produces garbage transcripts.
Wrong ref_text → F5 outputs short vocal artefacts instead of speech.

Fix: use F5's bundled clean reference (basic_ref_en.wav) with KNOWN transcript.
For user-uploaded voice: keep ref_text="" (whisper usually OK on clean audio).
"""
import sys
import os
import json
import base64
import io

print("[F5TTS-WORKER-VERSION: 2026-04-23-reftext-fix]", file=sys.stderr, flush=True)


# F5-TTS ships a clean reference in its package with this known transcript.
F5_BUNDLED_REFS = [
    ("/opt/f5tts-venv/lib/python3.12/site-packages/f5_tts/infer/examples/basic/basic_ref_en.wav",
     "Some call me nature, others call me mother nature."),
]

# Podcast-based fallback (30s cut from 22-min podcast — transcript unknown,
# use a generic English sentence that aligns reasonably)
PODCAST_REF = "/models/default_female_voice.wav"
PODCAST_REF_TEXT = "Hello everyone and welcome back to the show, today we have an amazing topic to discuss."


def find_reference(user_voice_path):
    """Pick best reference audio + known transcript.
    Priority: user voice > F5 bundled > podcast fallback."""
    if user_voice_path and os.path.exists(user_voice_path):
        print(f"[F5TTS] using USER voice: {user_voice_path}", file=sys.stderr)
        return user_voice_path, ""  # whisper on clean user audio usually OK

    for path, transcript in F5_BUNDLED_REFS:
        if os.path.exists(path):
            print(f"[F5TTS] using F5 BUNDLED ref: {path}", file=sys.stderr)
            print(f"[F5TTS] ref_text = {transcript!r}", file=sys.stderr)
            return path, transcript

    if os.path.exists(PODCAST_REF):
        print(f"[F5TTS] using PODCAST fallback: {PODCAST_REF}", file=sys.stderr)
        print(f"[F5TTS] ref_text (hardcoded guess) = {PODCAST_REF_TEXT!r}", file=sys.stderr)
        return PODCAST_REF, PODCAST_REF_TEXT

    print(f"[F5TTS] NO REFERENCE FOUND — generating without ref", file=sys.stderr)
    return None, ""


def main():
    request = json.loads(sys.stdin.read())
    text = request["text"]
    voice_sample_path = request.get("voice_sample_path")

    os.environ["HF_HOME"] = "/models/f5tts"

    # Redirect stdout to stderr during model loading (prevents torch/HF prints
    # from corrupting our final JSON stdout line)
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    import torch
    import torchaudio
    import numpy as np
    from f5_tts.api import F5TTS

    print(f"[F5TTS] torch={torch.__version__}, cuda={torch.cuda.is_available()}", file=sys.stderr)
    print(f"[F5TTS] gen_text received: {text!r} (len={len(text)})", file=sys.stderr, flush=True)

    ref_path, ref_text = find_reference(voice_sample_path)
    if ref_path and os.path.exists(ref_path):
        print(f"[F5TTS] ref size: {os.path.getsize(ref_path)} bytes", file=sys.stderr)

    model = F5TTS(device="cuda")
    print(f"[F5TTS] model loaded, running infer...", file=sys.stderr, flush=True)

    wav, sr, _ = model.infer(
        ref_file=ref_path,
        ref_text=ref_text,
        gen_text=text,
    )

    print(f"[F5TTS] infer returned type={type(wav).__name__} sr={sr}", file=sys.stderr, flush=True)
    if hasattr(wav, 'shape'):
        print(f"[F5TTS] wav.shape={wav.shape} duration={wav.shape[-1]/sr:.2f}s", file=sys.stderr, flush=True)

    if isinstance(wav, np.ndarray):
        wav_np = wav.squeeze()
    else:
        wav_np = wav.squeeze().cpu().numpy()

    print(f"[F5TTS] after squeeze: len={len(wav_np)} duration={len(wav_np)/sr:.2f}s "
          f"max_amp={np.abs(wav_np).max():.3f} mean_amp={np.abs(wav_np).mean():.4f}",
          file=sys.stderr, flush=True)

    # Trim trailing silence (threshold 0.005 — not so aggressive that it cuts whispery speech)
    threshold = 0.005
    abs_wav = np.abs(wav_np)
    above = np.where(abs_wav > threshold)[0]
    if len(above) > 0:
        end_sample = min(above[-1] + int(sr * 0.3), len(wav_np))
        wav_np = wav_np[:end_sample]
        print(f"[F5TTS] trimmed to duration={len(wav_np)/sr:.2f}s", file=sys.stderr, flush=True)
    else:
        print(f"[F5TTS] WARNING: entire wav below threshold {threshold} — silence!",
              file=sys.stderr, flush=True)

    # Add subtle room noise for natural sound
    room_noise = np.random.normal(0, 0.001, len(wav_np))
    wav_np = (wav_np + room_noise).clip(-1.0, 1.0).astype(np.float32)

    wav_final = torch.tensor(wav_np).unsqueeze(0)

    buf = io.BytesIO()
    torchaudio.save(buf, wav_final, sr, format="wav")
    buf.seek(0)
    audio_b64 = base64.b64encode(buf.read()).decode()

    print(f"[F5TTS] WAV base64: {len(audio_b64)} bytes", file=sys.stderr, flush=True)

    sys.stdout = real_stdout
    print(json.dumps({"status": "success", "audio": audio_b64}))


if __name__ == "__main__":
    main()
