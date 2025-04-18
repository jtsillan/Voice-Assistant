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


import speech_recognition as sr
from statemachine import StateMachine, State
from faster_whisper import WhisperModel
from perf_timer import PerformanceTimer
from python_variables import Bd
from Levenshtein import distance as lev_dis
import tempfile
import pyttsx3
import re
import time
import mysql.connector
import os
import threading


MICROPHONE_INDEX = 1
wake_word = "jorma"
model_size = "large-v2"
sample_rate = 16000
sample_width = 2
keyword_dict = {"soita": 1, "aika": 2, "lopeta": 3, "jaana": 4, "torres": 5}
command_word_list = {"kyllä": 6, "en halua": 7}

pf_timer = PerformanceTimer()


class VoiceAssistant(StateMachine):
    
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
                 

    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=sample_rate, device_index=MICROPHONE_INDEX)
        self.wake_word = wake_word
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        self.engine = pyttsx3.init()
        self.running = True
        self.active = True
        self.commands = keyword_dict
        self.command_word_list = command_word_list  
        self.unrecognized_attempts = 0
        self.contacts = None
        self.timeout_timer = None
        self.command_timeout = 5 
        self.voice = self.engine.setProperty('voice', self.engine.getProperty('voices')[1].id)

    def on_enter_state(self, event, state):
        """ Reset unrecognized attempts when returning to listening mode """
        print(f"on_enter_state() --> self: {self}, entering '{state.id}' state from '{event}' event.")
        print(f"on_enter_state() --> self.unrecognized_attempt before: {self.unrecognized_attempts}")
        self.unrecognized_attempts = 0
        print(f"on_enter_state() --> self.unrecognized_attempt after: {self.unrecognized_attempts}")

    def start_timeout(self):
        """Start a timeout based on the current state."""
        print(f"start_timeout() -->")
        if self.timeout_timer:
            self.timeout_timer.cancel()

        if self.current_state in [
            VoiceAssistant.waiting_for_command,
            VoiceAssistant.asking_contact_name,
            VoiceAssistant.asking_command_word,
        ]:
            self.timeout_timer = threading.Timer(self.command_timeout, self.handle_state_timeout)
            self.timeout_timer.start()

    def cancel_timeout(self):
        """Cancel any active timeout."""
        print(f"cancel_timeout() -->")
        if self.timeout_timer:
            self.timeout_timer.cancel()
            self.timeout_timer = None

    def handle_state_timeout(self):
        """Handles timeout when no response is received in a state."""
        print(f"handle_state_timeout() -->")
        if self.current_state == VoiceAssistant.waiting_for_command:
            #self.reset_from_waiting_command()
            self.response("No command received. Returning to listening mode.")
        elif self.current_state == VoiceAssistant.asking_contact_name:
            #self.reset_from_asking_contact_name()
            self.response("No contact name received. Returning to listening mode.")
        elif self.current_state == VoiceAssistant.asking_command_word:
            #self.reset_from_asking_command_word()
            self.response("No response received. Returning to listening mode.")
        
        self.cancel_timeout()    
        VoiceAssistant.reset()
        #self.unrecognized_attempts = 0  # Reset attempts

    def response(self, text):
        print(f"response() --> text: {text}")
        self.active = False
        self.engine.say(text)
        self.engine.runAndWait()
        self.active = True 
        self.start_timeout()

    def listen_audio(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("listen_audio() --> Listening for audio...")
            print(f"listen_audio() --> self.current_state: {self.current_state}")
            print(f"listen_audio() --> self.current_state_value: {self.current_state_value}")
            print(f"listen_audio() --> self.start_value: {self.start_value}")
            print(f"listen_audio() --> State.id: {State.id}")
            
            while self.running:
                #if VoiceAssistant. == 'Listening' and not self.active:
                # VoiceAssistant.STATE --> state to refer
                if self.current_state == VoiceAssistant.listening and not self.active:
                    continue
                try:
                    print("listen_audio() --> Recording...")
                    audio = self.recognizer.listen(source, timeout=None)
                    pf_timer.start()
                    self.process_audio(audio)
                except Exception as e:
                    print(f"listen_audio() --> Error processing audio: {e}")
                
    def process_audio(self, audio):
        print(f"process_audio() --> self.unrecognized_attempts: {self.unrecognized_attempts}")
        self.cancel_timeout()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio.get_wav_data(convert_rate=sample_rate, convert_width=sample_width))
            temp_audio_path = temp_audio.name

        segments, _ = self.model.transcribe(temp_audio_path, language="fi", condition_on_previous_text=False)
        transcribed_text = " ".join(segment.text for segment in segments)
        print(f"process_audio() --> transcribed_text: {transcribed_text}")
        cleaned_text = re.sub(r'[^A-Za-z ÅÄÖåäö]+', '', transcribed_text).lower().strip()
        print(f"process_audio() --> cleaned_text: {cleaned_text}")
        pf_timer.stop()
        os.remove(temp_audio_path)
        
        self.handle_transcription(cleaned_text)

    def handle_transcription(self, text):
        self.cancel_timeout()
        print(f"handle_transcription() --> self.current_state: {self.current_state}")
        if not text:
            return
        
        if self.current_state == VoiceAssistant.listening:
            if self.find_keyword_with_tolerance(text, self.wake_word):
                VoiceAssistant.trigger_detected()
                self.response("Listening?")
        elif self.current_state == VoiceAssistant.waiting_for_command:
            for cmd, cmd_id in self.commands.items():
                if self.find_keyword_with_tolerance(text, cmd):
                    VoiceAssistant.command_received()
                    self.execute_command(cmd_id)
                    return
            self.handle_unrecognized_input()
        elif self.current_state == VoiceAssistant.asking_contact_name:
            self.handle_contact_name(text)
        elif self.current_state == VoiceAssistant.asking_command_word:
            self.handle_command_word(text)
    
    def handle_unrecognized_input(self):
        print("handle_unrecognized_input()")
        self.unrecognized_attempts += 1
        if self.unrecognized_attempts >= 2:
            self.response("Command not recognized. Returning to listening mode.")
            VoiceAssistant.reset()
            self.unrecognized_attempts = 0
        else:
            self.response("Command not recognized, please repeat.")
    
    def handle_contact_name(self, text):
        print(f"handle_contact_name() --> text: {text}")
        for contact in self.contacts:
            if self.find_keyword_with_tolerance(text, contact["name"], threshold=2):
                self.response(f"Found contact {contact['name']}, would you like to call?")
                VoiceAssistant.request_command_word()
                return
        self.handle_unrecognized_input()
    
    def handle_command_word(self, text):
        print(f"handle_command_word() --> text: {text}")
        for word, word_id in self.command_word_list.items():
            if self.find_keyword_with_tolerance(text, word):
                self.execute_command(word_id)
                return
        self.handle_unrecognized_input()

    def find_keyword_with_tolerance(self, text, keyword, threshold=1):
        print(f"find_keyword_with_tolerance() --> text: {text}, keyword: {keyword}")
        return lev_dis(text, keyword.lower()) <= threshold

    def execute_command(self, command_id):
        print(f"execute_command() --> command_id: {command_id}")
        if command_id == 1:
            VoiceAssistant.request_contact()
            self.contacts = self.search_for_contacts()
            self.response("Who would you like to call?")
            return
        elif command_id == 2:
            print(f"execute_command() --> Kello on: {time.strftime('%H:%M:%S')}")
            self.response(f"Time is {time.strftime('%H:%M:%S')}")
        elif command_id == 3:
            print("execute_command() --> Exit command received. Stopping...")
            self.response("Stopping. Goodbye!")
            self.running = False
        elif command_id == 4:
            print("execute_command() --> Jaana on paras")
            self.response("Jaana is strong climber and makes great food!")
        elif command_id == 5:
            print(f"execute_command() --> Alkoholia koneeseen")
            self.response("Ari and Vito are coming for drinks.")
        elif command_id == 6:
            self.response("Calling now...")
            VoiceAssistant.initiate_call()
            time.sleep(5)
            self.response("Call ended.")
            self.unrecognized_attempts = 0
        elif command_id == 7:
            self.response("Call canceled.")
            self.unrecognized_attempts = 0
        VoiceAssistant.reset()

    def search_for_contacts(self):   
        print(f"search_for_contacts() --> ")   
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
                for contact in contacts:
                    print(f"search_for_contacts() --> contact: {contact}")
                    print(f"search_for_contacts() --> {contact['id']}, {contact['contact_name']}, {contact['phone_number']}")                    
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
