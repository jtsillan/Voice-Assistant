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


import graphviz

# Create a state diagram
dot = graphviz.Digraph("VoiceAssistantStateMachine", filename="images/voice_assistant", format="png")

# Define states
states = {
    "listening": "Listening",
    "waiting_for_command": "Waiting for command",
    "processing_command": "Processing command",
    "asking_contact_name": "Asking for contact name",
    "asking_command_word": "Asking for command word",
    "in_call": "In a call",
}

# Add states to the graph
for state_key, state_label in states.items():
    dot.node(state_key, state_label, shape="ellipse")

# Define transitions
transitions = [
    ("listening", "waiting_for_command", "Trigger detected"),
    ("waiting_for_command", "processing_command", "Command received"),
    ("processing_command", "asking_contact_name", "Request contact"),
    ("asking_contact_name", "asking_command_word", "Request command word"),
    ("asking_command_word", "in_call", "Initiate call"),
]

# Reset transitions
reset_transitions = [
    "waiting_for_command",
    "processing_command",
    "asking_contact_name",
    "asking_command_word",
    "in_call",
]

# Add transitions to the graph
for from_state, to_state, label in transitions:
    dot.edge(from_state, to_state, label)

# Add reset transitions back to "listening"
for state in reset_transitions:
    dot.edge(state, "listening", "Reset")

# Render and display the graph
dot.view()
