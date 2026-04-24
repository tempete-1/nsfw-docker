"""F5-TTS voice worker. Runs in isolated venv via subprocess.

Fix 2026-04-23 for 'кхм'/male-voice bugs:
- Use user's podcast voice (FEMALE) as default reference
- Reference is transcribed at BUILD time by whisper → saved to .txt file
- Worker reads the pre-computed transcript instead of relying on F5's
  runtime whisper-roulette (which produces garbage on noisy intros)
- F5 bundled basic_ref_en.wav is MALE — kept only as last-resort fallback
"""
import sys
import os
import json
import base64
import io

print("[F5TTS-WORKER-VERSION: 2026-04-23-female-ref-with-transcript]", file=sys.stderr, flush=True)


# Our own female voice (from user's podcast) — transcribed at build time
PODCAST_REF_WAV = "/models/default_female_voice.wav"
PODCAST_REF_TXT = "/models/default_female_voice.txt"

# F5's bundled MALE English reference — only as last-resort fallback
F5_BUNDLED_REF = "/opt/f5tts-venv/lib/python3.12/site-packages/f5_tts/infer/examples/basic/basic_ref_en.wav"
F5_BUNDLED_TEXT = "Some call me nature, others call me mother nature."


def find_reference(user_voice_path):
    """Pick best reference audio + transcript. Priority:
    1. User-uploaded voice (ref_text="" → F5's runtime whisper, usually OK on clean user audio)
    2. User's podcast (female) with pre-transcribed ref_text
    3. F5 bundled male fallback (last resort)
    """
    if user_voice_path and os.path.exists(user_voice_path):
        print(f"[F5TTS] using USER voice: {user_voice_path}", file=sys.stderr)
        return user_voice_path, ""

    if os.path.exists(PODCAST_REF_WAV) and os.path.exists(PODCAST_REF_TXT):
        with open(PODCAST_REF_TXT) as f:
            transcript = f.read().strip()
        if transcript:
            print(f"[F5TTS] using PODCAST FEMALE ref: {PODCAST_REF_WAV}", file=sys.stderr)
            print(f"[F5TTS] ref_text (from build-time whisper) = {transcript!r}", file=sys.stderr)
            return PODCAST_REF_WAV, transcript
        print(f"[F5TTS] podcast transcript file empty — falling through", file=sys.stderr)

    if os.path.exists(F5_BUNDLED_REF):
        print(f"[F5TTS] FALLBACK to F5 bundled MALE ref: {F5_BUNDLED_REF}", file=sys.stderr)
        return F5_BUNDLED_REF, F5_BUNDLED_TEXT

    print(f"[F5TTS] NO REFERENCE FOUND — generating without ref", file=sys.stderr)
    return None, ""


def main():
    request = json.loads(sys.stdin.read())
    text = request["text"]
    voice_sample_path = request.get("voice_sample_path")

    os.environ["HF_HOME"] = "/models/f5tts"

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

    # Quality params:
    # - nfe_step=48 (default 32): higher = better quality, slower. 48 is a good trade-off.
    # - remove_silence=True: drops silent gaps, reduces AI-like feel
    # - cfg_strength=2.0 (default 2.0): classifier-free guidance strength
    # - speed=1.0: normal pace
    infer_kwargs = dict(
        ref_file=ref_path,
        ref_text=ref_text,
        gen_text=text,
    )
    # Try to pass quality params — older F5 versions may not support them
    for key, val in (("nfe_step", 48), ("remove_silence", True), ("speed", 1.0)):
        try:
            import inspect
            sig = inspect.signature(model.infer)
            if key in sig.parameters:
                infer_kwargs[key] = val
        except Exception:
            pass

    print(f"[F5TTS] infer kwargs: {list(infer_kwargs.keys())}", file=sys.stderr, flush=True)
    wav, sr, _ = model.infer(**infer_kwargs)

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

    # Trim trailing silence (gentle threshold — don't cut quiet speech)
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

    # Very subtle room noise to mask digital artifacts
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
