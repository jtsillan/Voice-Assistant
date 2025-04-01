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
