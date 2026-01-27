import time
import os
from app import detect_stego

# Test files
audio_file = "static/aa-chal-ke-tujhe.mp3"
stego_audio_file = "Stego/stego_aa-chal-ke-tujhe.wav"

print("Testing detection speed on audio files...")

# Test clean audio
if os.path.exists(audio_file):
    start_time = time.time()
    result = detect_stego(audio_file)
    end_time = time.time()
    print(f"Clean audio detection: {result} (time: {end_time - start_time:.2f}s)")
else:
    print("Clean audio file not found")

# Test stego audio
if os.path.exists(stego_audio_file):
    start_time = time.time()
    result = detect_stego(stego_audio_file)
    end_time = time.time()
    print(f"Stego audio detection: {result} (time: {end_time - start_time:.2f}s)")
else:
    print("Stego audio file not found")

print("Testing complete.")
