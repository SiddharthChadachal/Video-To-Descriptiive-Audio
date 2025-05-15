import os
from pydub import AudioSegment
import moviepy.editor as mp

# Step 1: Function to extract audio from the video
def extract_audio_from_video(video_path, audio_output_path):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_output_path)
    print(f"Extracted audio saved to {audio_output_path}")

# Step 2: Function to extract timestamp from the audio filename (e.g., 'audio_5.32.wav' -> '5.32')
def extract_timestamp_from_filename(filename):
    try:
        return filename.split('_')[1].split('.')[0]
    except IndexError:
        return None

# Step 3: Function to generate a new video with pauses in the original video and inserted audio segments
def create_modified_video_with_audio(video_path, original_audio_path, audio_folder, output_video_path):
    original_audio = AudioSegment.from_file(original_audio_path)
    video = mp.VideoFileClip(video_path)
    new_video_clips = []
    last_timestamp = 0

    for filename in sorted(os.listdir(audio_folder)):  # Sort filenames to handle inserts in order
        if filename.endswith('.wav'):
            timestamp_str = extract_timestamp_from_filename(filename)

            if timestamp_str:
                timestamp_ms = float(timestamp_str) * 1000
                timestamp_seconds = float(timestamp_str)
                audio_file = os.path.join(audio_folder, filename)
                new_audio = AudioSegment.from_file(audio_file)
                new_audio_duration = len(new_audio) / 1000  # Convert from ms to seconds

                # Get video clips before the new audio insertion point
                if timestamp_seconds > last_timestamp:
                    pre_audio_clip = video.subclip(last_timestamp, timestamp_seconds)
                    new_video_clips.append(pre_audio_clip)

                # Freeze the video at the timestamp while the new audio plays
                frozen_frame = video.subclip(timestamp_seconds).get_frame(timestamp_seconds)
                frozen_clip = mp.ImageClip(frozen_frame).set_duration(new_audio_duration)
                new_video_clips.append(frozen_clip.set_audio(mp.AudioFileClip(audio_file)))

                # Update last timestamp to continue from the end of inserted audio
                last_timestamp = timestamp_seconds + new_audio_duration

    # Add remaining video after the last inserted audio
    if last_timestamp < video.duration:
        remaining_clip = video.subclip(last_timestamp, video.duration)
        new_video_clips.append(remaining_clip)

    # Concatenate all video clips
    final_video = mp.concatenate_videoclips(new_video_clips)

    # Export the final video with modified audio
    final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    print(f"Final video saved to {output_video_path}")

# Example usage
video_path = 'WhatsApp Video 2025-05-14 at 7.28.46 PM.mp4'
extracted_audio_path = 'extracted_audio.wav'
output_video_path = 'final_video_with_audio2.mp4'
audio_folder = '/content/audio_output'

# Step 1: Extract the original audio from the video
extract_audio_from_video(video_path, extracted_audio_path)

# Step 2: Create the modified video with pauses for inserted audio
create_modified_video_with_audio(video_path, extracted_audio_path, audio_folder, output_video_path)
