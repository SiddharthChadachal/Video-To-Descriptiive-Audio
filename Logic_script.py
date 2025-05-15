import cv2
import os
import torch
import soundfile as sf
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from deepface import DeepFace
from scenedetect import open_video, SceneManager, ContentDetector
from ultralytics import YOLO
from collections import Counter

# Function to check if the image is blurry
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

# Function to load YOLOv8 model
def load_yolov8_model():
    model = YOLO('yolov8n.pt')  # Use the correct YOLOv8 model file
    return model

# Function to detect objects using YOLOv8
def detect_objects_yolo(model, frame):
    results = model(frame)  # This returns a list of detections
    detected_classes = []

    for detection in results[0].boxes:
        class_id = int(detection.cls[0])
        class_name = results[0].names[class_id]
        detected_classes.append(class_name)

    return detected_classes

# Function to extract scenes, save frames with timestamps, and generate audio with timestamps
def extract_scenes_with_yolo(video_path, output_dir, audio_folder, threshold=30, blur_threshold=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)

    # Load YOLOv8 model
    yolo_model = load_yolov8_model()

    # Perform scene detection
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(frame_source=video)
    scene_list = scene_manager.get_scene_list()

    print(f"Detected {len(scene_list)} scenes.")

    # Open the video again for frame extraction
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video

    for i, (start, end) in enumerate(scene_list):
        start_frame = start.get_frames()
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = video_capture.read()

        if ret and frame is not None:
            if not is_blurry(frame, blur_threshold):
                # Detect objects using YOLOv8
                detected_objects = detect_objects_yolo(yolo_model, frame)

                if detected_objects:  # Check if objects were detected
                    # Calculate timestamp
                    timestamp = start_frame / fps
                    timestamp_str = "{:.2f}".format(timestamp)

                    # Save the frame with timestamp
                    frame_filename = f"frame_{timestamp_str}_start.jpg"
                    cv2.imwrite(f"{output_dir}/{frame_filename}", frame)
                    print(f"Extracted and processed frame for scene {i} at timestamp {timestamp_str} seconds")

                    # Generate caption, analyze expression, and save audio with timestamp
                    process_image_for_caption_and_audio(f"{output_dir}/{frame_filename}", detected_objects, audio_folder, timestamp_str)
                else:
                    print(f"No objects detected at {start_frame}, skipped.")
            else:
                print(f"Frame at {start_frame} is blurry, skipped.")
        else:
            print(f"Failed to read frame at {start_frame}")

    video_capture.release()

# Function to generate a caption for an image
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    inputs = caption_processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)
    return caption

# Function to analyze the expression in an image
def analyze_expression(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'])
        dominant_emotion = result[0]['dominant_emotion']
        return dominant_emotion
    except Exception as e:
        print(f"Error analyzing expression for {image_path}: {e}")
        return None

# Function to synthesize speech from text and save it as an audio file
def synthesize_speech(text, output_path):
    if not text:
        return

    description = "A male speaker with a low-pitched voice delivers his words in a monotonous manner with clear audio quality."

    input_ids = tts_tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tts_tokenizer(text, return_tensors="pt").input_ids.to(device)

    generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(output_path, audio_arr, tts_model.config.sampling_rate)

# Function to process an image to generate caption, analyze expressions, and create audio
def process_image_for_caption_and_audio(image_path, detected_objects, audio_folder, timestamp_str):
    print(f"Processing image: {image_path}")

    # Count detected objects
    object_counts = Counter(detected_objects)
    objects_description = ', '.join(f"{count} {obj}s" if count > 1 else obj for obj, count in object_counts.items())

    # Generate visual description and emotion analysis
    caption = generate_caption(image_path)
    emotion = analyze_expression(image_path)

    if caption and emotion:
        # Merge YOLO and caption processor output
        final_caption = f"The scene contains {objects_description}. {caption} The person in the image appears to be {emotion}."
        print(f"Final Caption: {final_caption}")

        # Synthesize audio
        audio_filename = f"audio_{timestamp_str}.wav"
        audio_path = os.path.join(audio_folder, audio_filename)
        synthesize_speech(final_caption, audio_path)

        print(f"Audio saved to: {audio_path}")
    else:
        print(f"Failed to generate caption or analyze expression for {image_path}")

# Load the Blip processor and model for image captioning
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the TTS processor and model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
tts_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

# Example usage
video_path = 'WhatsApp Video 2025-05-14 at 7.28.46 PM.mp4'
output_dir = '/content/output'
audio_folder = '/content/audio_output'
extract_scenes_with_yolo(video_path, output_dir, audio_folder, threshold=30, blur_threshold=100)
