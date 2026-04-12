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
    exaggeration = request.get("exaggeration", 0.7)
    voice_sample_path = request.get("voice_sample_path")

    os.environ["HF_HOME"] = "/models/chatterbox"

    # Redirect stdout to stderr during model loading to prevent library
    # warnings (e.g. torch weights_only) from corrupting JSON output
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    import torch
    import torchaudio
    from chatterbox.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(device="cuda")

    wav = model.generate(
        text=text,
        audio_prompt_path=voice_sample_path,
        exaggeration=exaggeration,
    )

    buf = io.BytesIO()
    torchaudio.save(buf, wav, model.sr, format="wav")
    buf.seek(0)
    audio_b64 = base64.b64encode(buf.read()).decode()

    # Restore stdout for JSON result only
    sys.stdout = real_stdout
    print(json.dumps({"status": "success", "audio": audio_b64}))


if __name__ == "__main__":
    main()
