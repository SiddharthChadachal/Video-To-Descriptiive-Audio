#!/bin/bash
echo " Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio  # For PyTorch and YOLOv5
pip install transformers  # For text generation
pip install gtts  # For Google Text-to-Speech
pip install opencv-python  # For video processing
pip install scenedetect
pip install --upgrade scenedetect
pip install git+https://github.com/huggingface/parler-tts.git
pip install transformers
pip install --upgrade protobuf
pip install deepface
pip install ultralytics
from ultralytics import YOLO
pip install moviepy pydub
echo " All dependencies installed successfully."
echo " To activate the virtual environment later, run: source venv/bin/activate"