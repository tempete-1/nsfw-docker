"""Fish Speech S2-Pro voice worker. Runs in isolated venv via subprocess.

Starts the Fish Speech API server, sends a TTS request via msgpack,
returns base64 WAV audio.
"""
import sys
import os
import json
import base64
import subprocess
import time
import urllib.request

sys.path.insert(0, "/opt/fish-speech")

CHECKPOINT_PATH = "/opt/fish-speech/checkpoints/s2-pro"
API_PORT = 8891
API_URL = f"http://127.0.0.1:{API_PORT}"


def wait_for_server(timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{API_URL}/v1/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def add_emotions(text: str) -> str:
    """Add Fish Speech inline emotion tags based on punctuation."""
    import re
    text = re.sub(r'\.\.\.', ' [breath]...', text)
    text = re.sub(r'—', ' [pause]—', text)
    text = re.sub(r'!\s', ' [excited] ', text)
    text = re.sub(r'\?\s', ' [curious] ', text)
    text = re.sub(r'!$', ' [excited]', text)
    text = re.sub(r'\?$', ' [curious]', text)
    text = re.sub(r'[Hh]aha|[Ll]ol|[Hh]ehe', '[laugh]', text)
    text = re.sub(r'[Mm]mm+', '[sigh] mmm', text)
    return text


def main():
    request = json.loads(sys.stdin.read())
    text = request["text"]
    voice_sample_path = request.get("voice_sample_path")
    temperature = request.get("temperature", 0.9)
    top_p = request.get("top_p", 0.9)
    repetition_penalty = request.get("repetition_penalty", 1.05)

    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    text = add_emotions(text)
    print(f"[FISH] Text with emotions: {repr(text[:200])}", file=sys.stderr, flush=True)

    DEFAULT_VOICE = "/models/default_female_voice.wav"
    if not voice_sample_path or not os.path.exists(voice_sample_path):
        voice_sample_path = DEFAULT_VOICE

    if voice_sample_path and os.path.exists(voice_sample_path):
        print(f"[FISH] Using voice sample: {voice_sample_path}", file=sys.stderr, flush=True)
    else:
        print(f"[FISH] WARNING: no voice sample found", file=sys.stderr, flush=True)
        voice_sample_path = None

    print(f"[FISH] Params: temp={temperature}, top_p={top_p}, rep={repetition_penalty}", file=sys.stderr, flush=True)

    # Start API server
    print(f"[FISH] Starting API server...", file=sys.stderr, flush=True)
    server = subprocess.Popen(
        [
            sys.executable, "tools/api_server.py",
            "--listen", f"127.0.0.1:{API_PORT}",
            "--llama-checkpoint-path", CHECKPOINT_PATH,
            "--decoder-checkpoint-path", os.path.join(CHECKPOINT_PATH, "codec.pth"),
            "--decoder-config-name", "modded_dac_vq",
            "--device", "cuda",
            "--half",
        ],
        cwd="/opt/fish-speech",
        stdout=sys.stderr,
        stderr=sys.stderr,
        env={**os.environ, "PYTHONPATH": "/opt/fish-speech", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    try:
        if not wait_for_server():
            raise RuntimeError("Fish Speech API server failed to start within 120s")
        print(f"[FISH] API server ready", file=sys.stderr, flush=True)

        # Build TTS request using msgpack (the format Fish Speech expects)
        import ormsgpack
        from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

        references = []
        if voice_sample_path:
            with open(voice_sample_path, "rb") as f:
                audio_bytes = f.read()
            references.append(ServeReferenceAudio(audio=audio_bytes, text=""))

        tts_request = ServeTTSRequest(
            text=text,
            references=references,
            reference_id=None,
            format="wav",
            streaming=False,
            max_new_tokens=2048,
            chunk_length=300,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )

        payload = ormsgpack.packb(tts_request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC)

        req = urllib.request.Request(
            f"{API_URL}/v1/tts",
            data=payload,
            headers={"Content-Type": "application/msgpack"},
        )

        resp = urllib.request.urlopen(req, timeout=300)
        audio_data = resp.read()

        audio_b64 = base64.b64encode(audio_data).decode()
        print(f"[FISH] Generated audio: {len(audio_data)} bytes", file=sys.stderr, flush=True)

    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except Exception:
            server.kill()

    sys.stdout = real_stdout
    print(json.dumps({"status": "success", "audio": audio_b64}))


if __name__ == "__main__":
    main()
