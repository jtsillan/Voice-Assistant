import queue
import sounddevice as sd
import librosa
import numpy as np
import torch
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, load_silero_vad

# Configuration
SAMPLE_RATE = 16000  # Whisper model works best with 16kHz
BUFFER_SIZE = 4096

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Faster-Whisper model
model = WhisperModel("large-v2", compute_type="int8")

# Load Silero VAD model on GPU if available
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)


(get_speech_timestamps, _, _, _, _) = utils

audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    """Callback function to store audio data in a queue."""
    if status:
        print(status, flush=True)
    audio_queue.put(indata.copy())

def is_speech(audio_data):
    """Check if the audio contains speech using Silero VAD."""
    audio_data = torch.from_numpy(audio_data.flatten()).float().to(device)
    speech_timestamps = get_speech_timestamps(audio_data, vad_model, sampling_rate=SAMPLE_RATE)
    return len(speech_timestamps) > 0

def process_audio():
    """Continuously process and transcribe audio."""
    audio_buffer = []
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=BUFFER_SIZE):
        print("Listening...")
        while True:
            data = audio_queue.get()
            
            # Check if data contains speech
            if is_speech(data):
                audio_buffer.append(data)
            
            if audio_buffer:
                # Convert buffer to numpy array
                audio_data = np.concatenate(audio_buffer, axis=0)
                
                # Convert audio to float32 and resample with librosa
                audio_data = librosa.resample(audio_data.flatten().astype(np.float32), orig_sr=SAMPLE_RATE, target_sr=SAMPLE_RATE)
                
                # Transcribe using Faster-Whisper
                segments, _ = model.transcribe(audio_data, language="fi")
                
                for segment in segments:
                    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
                
                # Clear buffer after processing
                audio_buffer.clear()

if __name__ == "__main__":
    process_audio()
