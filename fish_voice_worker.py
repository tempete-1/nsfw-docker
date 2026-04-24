"""Fish Speech S2 voice worker. Runs in isolated venv via subprocess."""
import sys
import os
import json
import base64
import io


def main():
    request = json.loads(sys.stdin.read())
    text = request["text"]
    voice_sample_path = request.get("voice_sample_path")

    os.environ["HF_HOME"] = "/models/fish-speech"

    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    import torch
    import torchaudio
    import numpy as np
    from fish_speech.inference import load_model, inference

    print(f"[FISH] Loading model...", file=sys.stderr, flush=True)

    # Load Fish Speech S2 model
    model = load_model(
        checkpoint_path="/models/fish-speech/s2",
        device="cuda",
    )

    # Use default female voice if none provided
    DEFAULT_VOICE = "/models/default_female_voice.wav"
    if not voice_sample_path or not os.path.exists(voice_sample_path):
        voice_sample_path = DEFAULT_VOICE

    if os.path.exists(voice_sample_path):
        print(f"[FISH] Using voice sample: {voice_sample_path}", file=sys.stderr, flush=True)
    else:
        print(f"[FISH] WARNING: voice sample not found: {voice_sample_path}", file=sys.stderr, flush=True)
        voice_sample_path = None

    print(f"[FISH] Generating: {repr(text[:100])}", file=sys.stderr, flush=True)

    # Generate audio with Fish Speech S2
    wav, sr = inference(
        model=model,
        text=text,
        ref_audio=voice_sample_path,
        device="cuda",
    )

    print(f"[FISH] Generated: duration={len(wav)/sr:.2f}s sr={sr}", file=sys.stderr, flush=True)

    if isinstance(wav, np.ndarray):
        wav_np = wav.squeeze()
    else:
        wav_np = wav.squeeze().cpu().numpy()

    # Trim trailing silence
    threshold = 0.01
    abs_wav = np.abs(wav_np)
    above = np.where(abs_wav > threshold)[0]
    if len(above) > 0:
        end_sample = min(above[-1] + int(sr * 0.3), len(wav_np))
        wav_np = wav_np[:end_sample]

    wav_np = np.clip(wav_np, -1.0, 1.0).astype(np.float32)
    wav_final = torch.tensor(wav_np).unsqueeze(0)

    buf = io.BytesIO()
    torchaudio.save(buf, wav_final, sr, format="wav")
    buf.seek(0)
    audio_b64 = base64.b64encode(buf.read()).decode()

    sys.stdout = real_stdout
    print(json.dumps({"status": "success", "audio": audio_b64}))


if __name__ == "__main__":
    main()
