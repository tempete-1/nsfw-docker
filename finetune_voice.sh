#!/bin/bash
# Fine-tune Fish Speech openaudio-s1-mini on custom voice data
# Run on GPU Pod with 24GB+ VRAM (RTX 4090, A40, etc.)
set -e

VENV="/opt/fish-speech-venv"
FISH="/opt/fish-speech"
CHECKPOINT="$FISH/checkpoints/openaudio-s1-mini"
DATA_DIR="/workspace/voice_finetune/data"
OUTPUT_DIR="/workspace/voice_finetune/results"
MERGED_DIR="$FISH/checkpoints/custom-voice"

echo "=== Step 0: Check prerequisites ==="
if [ ! -f "$CHECKPOINT/codec.pth" ]; then
    echo "ERROR: openaudio-s1-mini checkpoint not found at $CHECKPOINT"
    echo "Downloading..."
    $VENV/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('fishaudio/openaudio-s1-mini', local_dir='$CHECKPOINT')
"
fi

if [ ! -f "/models/voice_full.ogg" ]; then
    echo "ERROR: /models/voice_full.ogg not found"
    exit 1
fi

echo "=== Step 1: Prepare audio segments ==="
mkdir -p "$DATA_DIR/speaker1"

# Convert OGG to WAV
echo "Converting OGG to WAV..."
ffmpeg -i /models/voice_full.ogg -ar 44100 -ac 1 /tmp/voice_full.wav -y

# Install whisper for transcription
$VENV/bin/pip install faster-whisper 2>/dev/null || true

# Segment audio + transcribe using Python
echo "Segmenting and transcribing..."
$VENV/bin/python - <<'PYEOF'
import os
import subprocess
import json

DATA_DIR = "/workspace/voice_finetune/data/speaker1"
WAV_PATH = "/tmp/voice_full.wav"

# Use faster-whisper for segmentation + transcription
from faster_whisper import WhisperModel
print("Loading Whisper model (medium)...")
model = WhisperModel("medium", device="cuda", compute_type="float16")

print("Transcribing (this takes a few minutes)...")
segments, info = model.transcribe(WAV_PATH, language="en", word_timestamps=False)

count = 0
for seg in segments:
    start = seg.start
    end = seg.end
    text = seg.text.strip()

    if not text or len(text) < 5:
        continue

    duration = end - start
    if duration < 2.0 or duration > 30.0:
        continue

    fname = f"{start:.2f}-{end:.2f}"
    wav_out = os.path.join(DATA_DIR, f"{fname}.wav")
    lab_out = os.path.join(DATA_DIR, f"{fname}.lab")

    # Extract segment with ffmpeg
    subprocess.run([
        "ffmpeg", "-i", WAV_PATH,
        "-ss", str(start), "-t", str(duration),
        "-ar", "44100", "-ac", "1",
        "-y", wav_out
    ], capture_output=True)

    # Write transcription
    with open(lab_out, "w") as f:
        f.write(text)

    count += 1

print(f"Created {count} segments in {DATA_DIR}")
PYEOF

echo "=== Step 2: Loudness normalization ==="
$VENV/bin/pip install fish-audio-preprocess 2>/dev/null || true
$VENV/bin/python -m fish_audio_preprocess.cli loudness-norm "$DATA_DIR" --output "$DATA_DIR" 2>/dev/null || echo "Loudness norm skipped (optional)"

echo "=== Step 3: Extract semantic tokens ==="
cd "$FISH"
$VENV/bin/python tools/vqgan/extract_vq.py "$DATA_DIR" \
    --num-workers 1 --batch-size 16 \
    --config-name "modded_dac_vq" \
    --checkpoint-path "$CHECKPOINT/codec.pth"

echo "=== Step 4: Build dataset ==="
$VENV/bin/python tools/llama/build_dataset.py \
    --input "$DATA_DIR" \
    --output "$DATA_DIR/protos" \
    --text-extension .lab \
    --num-workers 4

echo "=== Step 5: LoRA fine-tune ==="
mkdir -p "$OUTPUT_DIR"
$VENV/bin/python fish_speech/train.py \
    --config-name text2semantic_finetune \
    pretrained_ckpt_path="$CHECKPOINT" \
    project="$OUTPUT_DIR" \
    "train_dataset.proto_files=[${DATA_DIR}/protos]" \
    "val_dataset.proto_files=[${DATA_DIR}/protos]" \
    +lora@model.model.lora_config=r_8_alpha_16

echo "=== Step 6: Merge LoRA weights ==="
# Find the latest checkpoint
LATEST_CKPT=$(ls -t "$OUTPUT_DIR/checkpoints/"*.ckpt 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found in $OUTPUT_DIR/checkpoints/"
    exit 1
fi
echo "Using checkpoint: $LATEST_CKPT"

$VENV/bin/python tools/llama/merge_lora.py \
    --lora-config r_8_alpha_16 \
    --base-weight "$CHECKPOINT" \
    --lora-weight "$LATEST_CKPT" \
    --output "$MERGED_DIR"

# Copy codec to merged dir
cp "$CHECKPOINT/codec.pth" "$MERGED_DIR/codec.pth"

echo ""
echo "=== DONE ==="
echo "Fine-tuned model saved to: $MERGED_DIR"
echo "Test with: $VENV/bin/python tools/api_server.py --llama-checkpoint-path $MERGED_DIR --decoder-checkpoint-path $MERGED_DIR/codec.pth --decoder-config-name modded_dac_vq --device cuda --half"
