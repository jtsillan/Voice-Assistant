"""
Copyright [2025] [Juha Sillanpää]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import torch
import pyaudio
import wave
import numpy as np
import tempfile
import os
import threading
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from huggingface_hub import hf_hub_download
from silero_vad import get_speech_timestamps
from perf_timer import PerformanceTimer

# Load the model and processor
MODEL_NAME = "Finnish-NLP/wav2vec2-xlsr-1b-finnish-lm-v2"
MODEL_DIR = "models"
sample_rate = 16000
os.makedirs(MODEL_DIR, exist_ok=True)

# Download and load the model and processor without using cache
processor_path = hf_hub_download(repo_id=MODEL_NAME, filename="preprocessor_config.json", cache_dir=MODEL_DIR)
model_path = hf_hub_download(repo_id=MODEL_NAME, filename="pytorch_model.bin", cache_dir=MODEL_DIR)
config_path = hf_hub_download(repo_id=MODEL_NAME, filename="config.json", cache_dir=MODEL_DIR)

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

pf_timer = PerformanceTimer()

"""
# Load VAD model
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                onnx=False,
                                force_reload=True,                                
                                trust_repo=True)
"""

#get_speech_timestamps = utils["get_speech_timestamps"]


def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=sample_rate, input=True,
                        frames_per_buffer=1024)
    print("Listening...")
    
    try:
        while True:
            frames = []
            for _ in range(int(sample_rate / 1024 * 1)):
                try:
                    data = stream.read(1024, exception_on_overflow=False)
                    frames.append(data)
                except IOError as e:                    
                    print(f"Audio stream error: {e}")
                    continue
                
            if not frames:
                continue
            
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).copy()
            
            if len(audio_data) == 0:
                continue
            
            process_audio(frames, audio)
    
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

def process_audio(frames, audio):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        temp_filename = temp_wav.name
        waveFile = wave.open(temp_filename, 'wb')
        waveFile.setnchannels(1)
        waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(16000)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        
    print("Transcription:", transcribe_audio(temp_filename))
    os.remove(temp_filename)

def transcribe_audio(filename):
    with wave.open(filename, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)
    
    input_values = processor(audio_data.astype(np.float32), sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

if __name__ == "__main__":
    record_audio()
