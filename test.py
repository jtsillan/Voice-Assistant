"""
MIT License

Copyright (c) 2025 Juha Sillanpää

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import sys
import os
from pathlib import Path


def set_cuda_paths():
    print("Set cuda paths...")
    venv_base = Path(sys.executable).parent.parent
    nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    cuda_path = nvidia_base_path / 'cuda_runtime' / 'bin'
    cublas_path = nvidia_base_path / 'cublas' / 'bin'
    cudnn_path = nvidia_base_path / 'cudnn' / 'bin'
    paths_to_add = [str(cuda_path), str(cublas_path), str(cudnn_path)]
    env_vars = ['CUDA_PATH', 'CUDA_PATH_V12_4', 'PATH']
    
    for env_var in env_vars:
        current_value = os.environ.get(env_var, '')
        new_value = os.pathsep.join(paths_to_add + [current_value] if current_value else paths_to_add)
        os.environ[env_var] = new_value

set_cuda_paths()



import torch
import queue
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

# Load Silero VAD model
model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True)
(get_speech_timestamps, _, _, _, _) = utils

# Load Whisper model (choose "small", "medium", or "large-v2" for better accuracy)
whisper_model = WhisperModel("large-v2", device="cuda", compute_type="int8")

# Audio stream settings
SAMPLE_RATE = 16000  # Required for Silero VAD
CHUNK_DURATION = 0.5  # Seconds per chunk
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # Samples per chunk

# Queue for streaming audio data
audio_queue = queue.Queue()

# Callback function for sounddevice stream
def callback(indata, frames, time, status):
    """Captures microphone input and adds it to the queue."""
    if status:
        print(f"Stream Error: {status}")
    audio_queue.put(indata.copy())

# Stream processing function
def process_audio_stream():
    """Processes live audio using Silero VAD and transcribes with Faster-Whisper."""
    print("Listening... (Press Ctrl+C to stop)")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        buffer = np.array([], dtype=np.float32)
        try:
            while True:
                # Get audio from queue
                chunk = audio_queue.get()
                buffer = np.append(buffer, chunk)

                # Ensure buffer is at least 1 second for analysis
                if len(buffer) >= SAMPLE_RATE:
                    buffer_tensor = torch.tensor(buffer, dtype=torch.float32)

                    # Apply VAD to detect speech
                    speech_timestamps = get_speech_timestamps(buffer_tensor, model, sampling_rate=SAMPLE_RATE)

                    if speech_timestamps:
                        # Extract only speech segments
                        speech_audio = torch.cat([buffer_tensor[t['start']:t['end']] for t in speech_timestamps])

                        # Convert speech tensor to NumPy array
                        speech_np = speech_audio.numpy()

                        # Save temp WAV file
                        temp_filename = "temp_audio.wav"
                        write(temp_filename, SAMPLE_RATE, (speech_np * 32767).astype(np.int16))

                        # Transcribe using Faster-Whisper
                        segments, _ = whisper_model.transcribe(temp_filename, language="fi")
                        for segment in segments:
                            print(f"Recognized: {segment.text}")

                    # Reset buffer
                    buffer = np.array([], dtype=np.float32)

        except KeyboardInterrupt:
            print("Stopping...")

# Run the real-time speech recognition
process_audio_stream()
