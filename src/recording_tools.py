import os
import cv2
import pyaudio
import wave
from datetime import datetime
import threading

class RecordVideo:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        from pathlib import Path

        # Get the directory where this script is located
        script_dir = Path(__file__).resolve().parent

        # Navigate up to the project root, then into 'data/recordings'
        base_dir = script_dir.parent / 'data' / 'recordings'

        self.base_dir = str(base_dir)
        self.video_filename = f'output_{self.timestamp}.mp4'
        self.audio_filename = f'output_{self.timestamp}.wav'
        self.frame_rate = 20
        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.sampling_rate = 44100
        self.recording = False

        # Ensure correct subdirectories exist
        video_dir = os.path.join(self.base_dir, 'video')
        audio_dir = os.path.join(self.base_dir, 'audio')
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)

        self.video_path = os.path.join(video_dir, self.video_filename)
        self.audio_path = os.path.join(audio_dir, self.audio_filename)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Unable to access the camera.")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.video_path,
            self.fourcc,
            self.frame_rate,
            (width, height)
        )

    def record_audio(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.sample_format,
            channels=self.channels,
            rate=self.sampling_rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        frames = []
        try:
            while self.recording:
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            with wave.open(self.audio_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(audio.get_sample_size(self.sample_format))
                wf.setframerate(self.sampling_rate)
                wf.writeframes(b''.join(frames))

def record():
    try:
        video_class = RecordVideo()
        video_class.recording = True
        audio_thread = threading.Thread(target=video_class.record_audio)
        audio_thread.start()

        print("Recording started. Press 'q' to stop recording.")

        while video_class.recording:
            ret, frame = video_class.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            video_class.video_writer.write(frame)
            cv2.imshow('Recording', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_class.recording = False

        audio_thread.join()
        print('Audio thread finished.')
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'video_class' in locals():
            if hasattr(video_class, 'cap'):
                video_class.cap.release()
            if hasattr(video_class, 'video_writer'):
                video_class.video_writer.release()
        cv2.destroyAllWindows()
        print(f"Recording finished. Video saved as {getattr(video_class, 'video_path', 'N/A')} and audio saved as {getattr(video_class, 'audio_path', 'N/A')}.")

if __name__ == "__main__":
    record()
 