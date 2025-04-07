# Voice Assistant
This repository is based on Bachelor's Thesis that uses large language models (LLM) in Voice Assistant functionalities in Finnish language. Uses faster_whisper backend.

Different files are as follows:

#### va_single_thread.py
This is main file using faster_whisper Whisper model in a single thread. GPU in large-v2 model takes about 0.9 seconds to transcribe Finnish speech to text.

#### va_single_thread_super.py
Same as above but inherits StateMachine model and states can be referred differently. Not completed.

#### va_wav2vec.py
Currently not working properly.

#### va_librosa.py
Testing how librosa is working with Whisper model.

#### state_diagram.py
Creates a state diagram from VoiceAssistant class