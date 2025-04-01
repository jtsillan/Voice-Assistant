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
    #print("Set cuda paths...")
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



import speech_recognition as sr
from time import sleep
from statemachine import StateMachine, State
from faster_whisper import WhisperModel
from perf_timer import PerformanceTimer
from Levenshtein import distance as lev_dis
from python_variables import Bd
from gtts import gTTS
from playsound import playsound
import tempfile
import pyttsx3
import re
import time
import mysql.connector
import os
import threading


# Check available models: https://github.com/openai/whisper
#model_size = "base"
#model_size = "small"
#model_size = "medium"
model_size = "large-v2"

MICROPHONE_INDEX = 1
sample_rate = 16000
sample_width = 2

wake_word = "pirjo"
keyword_dict = {"soita": 1, "aika": 2, "lopeta": 3, "mitä kuuluu": 4}
command_word_dict = {"kyllä": 6, "en": 7}

pf_timer = PerformanceTimer()


class VoiceAssistantStateMachine(StateMachine):
    listening = State("Listening", initial=True)
    waiting_for_command = State("Waiting for command")
    processing_command = State("Processing command")
    asking_contact_name = State("Asking for contact name")
    asking_command_word = State("Asking for command word")
    in_call = State("In a call")
    
    trigger_detected = listening.to(waiting_for_command)
    command_received = waiting_for_command.to(processing_command)
    request_contact = processing_command.to(asking_contact_name)
    request_command_word = asking_contact_name.to(asking_command_word)
    initiate_call = asking_command_word.to(in_call)
    
    reset = waiting_for_command.to(listening) | processing_command.to(listening) | \
                     asking_contact_name.to(listening) | asking_command_word.to(listening) | in_call.to(listening)    

    def on_enter_state(self, event, state):
        """ Reset unrecognized attempts when returning to listening mode """
        print(f"on_enter_state() --> Entering '{state.id}' state from '{event}' event.")
        #if hasattr(self, 'assistant'):
            #self.assistant.unrecognized_attempts = 0


class VoiceAssistant:
    def __init__(self):
        self.state_machine = VoiceAssistantStateMachine()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=sample_rate, device_index=MICROPHONE_INDEX)
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")        
        # CPU usage works really slow on bigger models (medium, large) and not so good accurancy on smaller (base, small)
        #self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.engine = pyttsx3.init()
        self.wake_word = wake_word
        self.running = True
        self.active = True
        self.commands = keyword_dict
        self.command_word_dict = command_word_dict  
        self.unrecognized_attempts = 0
        self.contacts = None
        self.timeout_timer = None
        self.command_timeout = 10 
        self.voice = self.engine.setProperty('voice', self.engine.getProperty('voices')[1].id)

    def start_timeout(self):
        """Start a timeout based on the current state."""
        if self.timeout_timer:
            self.timeout_timer.cancel()

        if self.state_machine.current_state in [
            self.state_machine.waiting_for_command,
            self.state_machine.asking_contact_name,
            self.state_machine.asking_command_word,
        ]:
            self.timeout_timer = threading.Timer(self.command_timeout, self.handle_state_timeout)
            self.timeout_timer.start()

    def cancel_timeout(self):
        """Cancel any active timeout."""
        if self.timeout_timer:
            self.timeout_timer.cancel()
            self.timeout_timer = None

    def handle_state_timeout(self):
        """Handles timeout when no response is received in a state."""
        if self.state_machine.current_state == self.state_machine.waiting_for_command:
            self.state_machine.reset()
            self.response("Käskyä ei vastaanotettu. Palataan kuuntelutilaan.")
        elif self.state_machine.current_state == self.state_machine.asking_contact_name:
            self.state_machine.reset()
            self.response("Kontaktia ei vastaanotettu. Palataan kuuntelutilaan.")
        elif self.state_machine.current_state == self.state_machine.asking_command_word:
            self.state_machine.reset()
            self.response("Komentoa ei vastaanotettu. Palataan kuuntelutilaan.")
        
        self.cancel_timeout()    
        # Reset attempts
        self.unrecognized_attempts = 0 

    def response(self, text):
        """
        Play audio responce using Google TTS engine. HTTPS connection needed.
        arguments: text (string)
        """
        self.active = False
        file_path = "play_text.mp3"
        my_text = gTTS(text=text, lang="fi")
        my_text.save(file_path)        
        playsound(file_path)
        os.remove("play_text.mp3")
        self.active = True 
        self.start_timeout()

    def listen_audio(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while self.running:
                if self.state_machine.current_state == self.state_machine.listening and not self.active:
                    continue
                try:
                    audio = self.recognizer.listen(source, timeout=None)
                    pf_timer.start()
                    self.process_audio(audio)
                except Exception as e:
                    print(f"listen_audio() --> Error processing audio: {e}")
                
    def process_audio(self, audio):
        self.cancel_timeout()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio.get_wav_data(convert_rate=sample_rate, convert_width=sample_width))
            temp_audio_path = temp_audio.name

        segments, _ = self.model.transcribe(temp_audio_path, language="fi", condition_on_previous_text=True)
        transcribed_text = " ".join(segment.text for segment in segments)
        cleaned_text = re.sub(r'[^A-Za-z ÅÄÖåäö]+', '', transcribed_text).lower().strip()
        print(f"Text: {cleaned_text}")
        pf_timer.stop()
        os.remove(temp_audio_path)
        
        self.handle_transcription(cleaned_text)

    def handle_transcription(self, text):
        self.cancel_timeout()
        if not text:
            return        
        if self.state_machine.current_state == self.state_machine.listening:
            if self.find_keyword_with_tolerance(text, self.wake_word):
                self.state_machine.trigger_detected()
                self.response("Kuuntelen")
        elif self.state_machine.current_state == self.state_machine.waiting_for_command:
            for cmd, cmd_id in self.commands.items():
                if self.find_keyword_with_tolerance(text, cmd):
                    self.state_machine.command_received()
                    self.execute_command(cmd_id)
                    return
            self.handle_unrecognized_input()
        elif self.state_machine.current_state == self.state_machine.asking_contact_name:
            self.handle_contact_name(text)
        elif self.state_machine.current_state == self.state_machine.asking_command_word:
            self.handle_command_word(text)
    
    def handle_unrecognized_input(self):
        self.unrecognized_attempts += 1
        if self.unrecognized_attempts >= 2:
            self.response("Komentoa ei tunnistettu. Palataan kuuntelutilaan.")
            self.state_machine.reset()
            self.unrecognized_attempts = 0
        else:
            self.response("Komentoa ei tunnistettu. Voisitko toistaa.")
    
    def handle_contact_name(self, text):
        for contact in self.contacts:
            if self.find_keyword_with_tolerance(text, contact["name"], threshold=1):
                self.response(f"Löytyi kontakti nimeltä {contact['name']}. Haluaisitko soittaa?")
                self.state_machine.request_command_word()
                return
        self.handle_unrecognized_input()
    
    def handle_command_word(self, text):
        for word, word_id in self.command_word_dict.items():
            if self.find_keyword_with_tolerance(text, word):
                self.execute_command(word_id)
                return
        self.handle_unrecognized_input()

    def find_keyword_with_tolerance(self, text: str, keyword: str, threshold=1):
        words = text.split()
        matches = [word for word in words if lev_dis(word.lower(), keyword.lower()) <= threshold]
        return matches

    def execute_command(self, command_id):
        if command_id == 1:
            self.state_machine.request_contact()
            self.contacts = self.search_for_contacts()
            self.response("Kenelle haluaisit soittaa?")
            return
        elif command_id == 2:
            self.response(f"Kello on {time.strftime('%H:%M')}")
        elif command_id == 3:
            self.response("Lopetan. Näkemiin!")
            self.running = False
        elif command_id == 4:
            self.response("Hyvää kuuluu!")
        elif command_id == 6:
            self.response("Soitan")
            self.state_machine.initiate_call()
            # Mimic making a call
            time.sleep(5)
            self.response("Puhelu loppui")
            self.unrecognized_attempts = 0
        elif command_id == 7:
            self.response("Puhelu peruttu")
            self.unrecognized_attempts = 0
        self.state_machine.reset()

    def search_for_contacts(self):   
        """
        """ 
        connection = None
        cursor = None        
        try:
            connection = mysql.connector.connect(user=Bd.user, host=Bd.host, database=Bd.database)
            cursor = connection.cursor(dictionary=True)
            query = "SELECT * FROM contacts AS c INNER JOIN users AS u ON c.users_id = u.id INNER JOIN phone_numbers AS pn ON c.phone_numbers_id = pn.id WHERE c.users_id = 1"
            cursor.execute(query)
            contacts = cursor.fetchall()
            info = []
            
            if self.contacts is not None:
                # clear old data
                self.contacts = []
            if contacts != []:
                # Add new data to list
                for contact in contacts:                   
                    info.append({"id": contact['id'], "name": contact['contact_name'], "number": contact['phone_number']})
                    
            return info   
        
        except Exception as ex:
            print(f"search_for_contacts() --> ex: {ex}")
            
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None and connection.is_connected():
                connection.close()  

    def run(self):
        self.listen_audio()


if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
