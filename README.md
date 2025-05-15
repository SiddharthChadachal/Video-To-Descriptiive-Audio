🎬 Video Description Generator with Emotion & Audio Narration
This project automatically transforms a video into an enriched audiovisual experience by detecting scenes, identifying objects, generating captions, analyzing emotions, and inserting descriptive audio narrations at relevant timestamps. It's especially useful for accessibility applications (e.g., aiding visually impaired viewers) or creating automated content summaries.

🔧 What It Does
Scene Detection & Frame Extraction
Uses scenedetect to find meaningful scene changes in the video and extract representative frames.

Object Detection
Applies YOLOv8 to identify objects within selected video frames.

Image Captioning
Uses the BLIP image captioning model to generate descriptive captions for each scene.

Emotion Analysis
Leverages DeepFace to detect the dominant facial emotion in each scene.

Text-to-Speech (TTS) Synthesis
Uses Parler-TTS to generate natural speech from the combined scene description and emotion.

Audio Insertion & Final Video Generation
Combines all generated audio clips with the original video by inserting scene-based pauses and descriptive audio.

🗂 Files Overview
📄 Logic_script.py
This script:

Detects scenes in the input video.

Filters out blurry frames.

Performs object detection (YOLOv8).

Generates captions (BLIP).

Analyzes expressions (DeepFace).

Synthesizes descriptive audio using Parler-TTS.

Saves audio clips timestamped to match the video.

📄 combine_script.py
This script:

Extracts the original audio from the video.

Inserts generated narration audio at the exact scene timestamps.

Freezes the video frame during narration to provide focus.

Reconstructs the final video with all modifications applied.

