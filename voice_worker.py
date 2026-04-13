"""Standalone Chatterbox TTS worker. Runs in isolated venv via subprocess."""
import sys
import os
import json
import base64
import io

def main():
    """Read JSON from stdin, generate voice, write base64 WAV to stdout."""
    request = json.loads(sys.stdin.read())
    text = request["text"]
    exaggeration = request.get("exaggeration", 0.3)
    voice_sample_path = request.get("voice_sample_path")

    os.environ["HF_HOME"] = "/models/chatterbox"

    # Redirect stdout to stderr during model loading to prevent library
    # warnings (e.g. torch weights_only) from corrupting JSON output
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    import torch
    # Chatterbox checkpoints contain non-tensor data (dicts, configs).
    # torch 2.6+ defaults to weights_only=True which breaks loading → garbled output.
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})

    import torchaudio
    from chatterbox.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(device="cuda")

    # Use default female voice sample if none provided
    DEFAULT_VOICE = "/models/default_female_voice.wav"
    if not voice_sample_path or not os.path.exists(voice_sample_path):
        voice_sample_path = DEFAULT_VOICE

    if os.path.exists(voice_sample_path):
        print(f"Using voice sample: {voice_sample_path} ({os.path.getsize(voice_sample_path)} bytes)", file=sys.stderr)
    else:
        print(f"WARNING: voice sample not found: {voice_sample_path}, generating without reference", file=sys.stderr)
        voice_sample_path = None

    wav = model.generate(
        text=text,
        audio_prompt_path=voice_sample_path,
        exaggeration=exaggeration,
        cfg_weight=0.7,
        repetition_penalty=1.5,
        temperature=0.6,
        top_p=0.95,
    )

    # Trim trailing silence/artifacts
    import numpy as np
    wav_np = wav.squeeze().cpu().numpy()
    # Find last sample above threshold (trim trailing noise)
    threshold = 0.01
    abs_wav = np.abs(wav_np)
    # Find last loud sample, then add 0.3s padding
    above = np.where(abs_wav > threshold)[0]
    if len(above) > 0:
        end_sample = min(above[-1] + int(model.sr * 0.3), len(wav_np))
        wav_np = wav_np[:end_sample]
    # Add subtle room noise to sound like real phone recording
    room_noise = np.random.normal(0, 0.001, len(wav_np))
    wav_np = wav_np + room_noise
    wav_np = np.clip(wav_np, -1.0, 1.0).astype(np.float32)

    wav_final = torch.tensor(wav_np).unsqueeze(0)

    buf = io.BytesIO()
    torchaudio.save(buf, wav_final, model.sr, format="wav")
    buf.seek(0)
    audio_b64 = base64.b64encode(buf.read()).decode()

    # Restore stdout for JSON result only
    sys.stdout = real_stdout
    print(json.dumps({"status": "success", "audio": audio_b64}))


if __name__ == "__main__":
    main()
