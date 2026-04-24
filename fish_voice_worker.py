"""Fish Speech S2-Pro voice worker. Runs in isolated venv via subprocess.

Uses the HTTP API server approach: starts api_server.py, sends TTS request,
returns base64 WAV audio.
"""
import sys
import os
import json
import base64
import io
import subprocess
import time
import urllib.request


CHECKPOINT_PATH = "/opt/fish-speech/checkpoints/s2-pro"
API_PORT = 8891
API_URL = f"http://127.0.0.1:{API_PORT}"


def wait_for_server(timeout=120):
    """Wait until Fish Speech API server is ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{API_URL}/v1/health", timeout=2)
            return True
        except Exception:
            try:
                urllib.request.urlopen(f"{API_URL}/docs", timeout=2)
                return True
            except Exception:
                time.sleep(1)
    return False


def main():
    request = json.loads(sys.stdin.read())
    text = request["text"]
    voice_sample_path = request.get("voice_sample_path")

    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    # Use default female voice if none provided
    DEFAULT_VOICE = "/models/default_female_voice.wav"
    if not voice_sample_path or not os.path.exists(voice_sample_path):
        voice_sample_path = DEFAULT_VOICE

    if os.path.exists(voice_sample_path):
        print(f"[FISH] Using voice sample: {voice_sample_path}", file=sys.stderr, flush=True)
    else:
        print(f"[FISH] WARNING: no voice sample found", file=sys.stderr, flush=True)
        voice_sample_path = None

    print(f"[FISH] Generating: {repr(text[:100])}", file=sys.stderr, flush=True)

    # Start API server
    print(f"[FISH] Starting API server...", file=sys.stderr, flush=True)
    server = subprocess.Popen(
        [
            sys.executable, "tools/api_server.py",
            "--listen", f"0.0.0.0:{API_PORT}",
            "--llama-checkpoint-path", CHECKPOINT_PATH,
            "--decoder-checkpoint-path", os.path.join(CHECKPOINT_PATH, "codec.pth"),
        ],
        cwd="/opt/fish-speech",
        stdout=sys.stderr,
        stderr=sys.stderr,
    )

    try:
        if not wait_for_server():
            raise RuntimeError("Fish Speech API server failed to start")
        print(f"[FISH] API server ready", file=sys.stderr, flush=True)

        # Build multipart request
        import mimetypes
        boundary = "----FishSpeechBoundary"
        body_parts = []

        # Add text field
        body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\n{text}")

        # Add reference audio if available
        if voice_sample_path and os.path.exists(voice_sample_path):
            with open(voice_sample_path, "rb") as f:
                audio_data = f.read()
            body_parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"reference_audio\"; filename=\"ref.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
                + audio_data
            )

        # Try the simple /v1/tts endpoint first
        tts_payload = json.dumps({
            "text": text,
            "reference_id": None,
            "normalize": True,
            "format": "wav",
            "streaming": False,
        }).encode()

        req = urllib.request.Request(
            f"{API_URL}/v1/tts",
            data=tts_payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            resp = urllib.request.urlopen(req, timeout=300)
            audio_bytes = resp.read()
        except Exception as e:
            print(f"[FISH] /v1/tts failed ({e}), trying CLI fallback...", file=sys.stderr, flush=True)
            # Fallback: use CLI inference directly
            server.terminate()
            server.wait()
            audio_bytes = cli_inference(text, voice_sample_path)

        audio_b64 = base64.b64encode(audio_bytes).decode()
        print(f"[FISH] Generated audio: {len(audio_bytes)} bytes", file=sys.stderr, flush=True)

    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except Exception:
            server.kill()

    sys.stdout = real_stdout
    print(json.dumps({"status": "success", "audio": audio_b64}))


def cli_inference(text, voice_sample_path):
    """Fallback: run Fish Speech inference via CLI tools."""
    import tempfile

    work_dir = tempfile.mkdtemp(prefix="fish_")
    output_wav = os.path.join(work_dir, "output.wav")

    cmd = [
        sys.executable, "tools/inference.py",
        "--text", text,
        "--checkpoint-path", CHECKPOINT_PATH,
        "--output", output_wav,
    ]
    if voice_sample_path:
        cmd.extend(["--reference-audio", voice_sample_path])

    result = subprocess.run(
        cmd,
        cwd="/opt/fish-speech",
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        print(f"[FISH] CLI stderr: {result.stderr[-500:]}", file=sys.stderr, flush=True)
        raise RuntimeError(f"Fish Speech CLI failed: {result.stderr[-300:]}")

    with open(output_wav, "rb") as f:
        return f.read()


if __name__ == "__main__":
    main()
