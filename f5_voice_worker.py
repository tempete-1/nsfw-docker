"""F5-TTS voice worker. Runs in isolated venv via subprocess."""
import sys
import os
import json
import base64
import io

print("[F5TTS-WORKER-VERSION-MARKER: 2026-04-21-v2-ndarray-fix]", file=sys.stderr, flush=True)

def main():
    request = json.loads(sys.stdin.read())
    text = request["text"]
    voice_sample_path = request.get("voice_sample_path")

    os.environ["HF_HOME"] = "/models/f5tts"

    # Redirect stdout to stderr during model loading
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    import torch
    import torchaudio
    import numpy as np
    from f5_tts.api import F5TTS

    model = F5TTS(device="cuda")

    # Use default female voice if none provided
    DEFAULT_VOICE = "/models/default_female_voice.wav"
    if not voice_sample_path or not os.path.exists(voice_sample_path):
        voice_sample_path = DEFAULT_VOICE

    if os.path.exists(voice_sample_path):
        print(f"Using voice sample: {voice_sample_path}", file=sys.stderr)
    else:
        print(f"WARNING: voice sample not found: {voice_sample_path}", file=sys.stderr)
        voice_sample_path = None

    print(f"[F5TTS-DEBUG] gen_text received: {repr(text)} (len={len(text)})", file=sys.stderr, flush=True)

    wav, sr, _ = model.infer(
        ref_file=voice_sample_path,
        ref_text="",
        gen_text=text,
    )

    print(f"[F5TTS-DEBUG] infer returned type={type(wav).__name__} sr={sr}", file=sys.stderr, flush=True)
    if hasattr(wav, 'shape'):
        print(f"[F5TTS-DEBUG] wav.shape={wav.shape} duration={wav.shape[-1]/sr:.2f}s", file=sys.stderr, flush=True)

    if isinstance(wav, np.ndarray):
        wav_np = wav.squeeze()
    else:
        wav_np = wav.squeeze().cpu().numpy()

    print(f"[F5TTS-DEBUG] after squeeze: len={len(wav_np)} duration={len(wav_np)/sr:.2f}s", file=sys.stderr, flush=True)

    # Trim trailing silence
    threshold = 0.01
    abs_wav = np.abs(wav_np)
    above = np.where(abs_wav > threshold)[0]
    if len(above) > 0:
        end_sample = min(above[-1] + int(sr * 0.3), len(wav_np))
        wav_np = wav_np[:end_sample]

    print(f"[F5TTS-DEBUG] after trim: len={len(wav_np)} duration={len(wav_np)/sr:.2f}s", file=sys.stderr, flush=True)

    # Add subtle room noise
    room_noise = np.random.normal(0, 0.001, len(wav_np))
    wav_np = (wav_np + room_noise).clip(-1.0, 1.0).astype(np.float32)

    wav_final = torch.tensor(wav_np).unsqueeze(0)

    buf = io.BytesIO()
    torchaudio.save(buf, wav_final, sr, format="wav")
    buf.seek(0)
    audio_b64 = base64.b64encode(buf.read()).decode()

    sys.stdout = real_stdout
    print(json.dumps({"status": "success", "audio": audio_b64}))


if __name__ == "__main__":
    main()
