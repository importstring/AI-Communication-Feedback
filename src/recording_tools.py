import os
import cv2
import pyaudio
import wave
from datetime import datetime
import threading

class RecordVideo:
    def __init__(self):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.base_dir = '../data/recordings'
        
        self.video_filename = f'output_{timestamp}.mp4'
        self.audio_filename = f'output_{timestamp}.wav'
        
        self.frame_rate = 60
        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.sampling_rate = 44100
        self.recording = False

        video_dir = os.path.join(self.base_dir, 'video')
        audio_dir = os.path.join(self.base_dir, 'audio')
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        
        self.video_path = os.path.join(video_dir, self.video_filename)
        self.audio_path = os.path.join(audio_dir, self.audio_filename)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Unable to access the camera.")
        
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        self.video_writer = cv2.VideoWriter(
            self.video_path, 
            self.fourcc, 
            self.frame_rate, 
            (640, 480)
        )

    def update_path(self, sub_dir='', filename=''):
        """
        Constructs a path by joining base_dir, sub_dir and filename.
        
        Args:
            sub_dir (str): Subdirectory name
            filename (str): Filename to append
            
        Returns:
            str: Complete file path
        """
        if sub_dir:
            if sub_dir.startswith('/'):
                sub_dir = sub_dir[1:]
            if sub_dir.endswith('/'):
                sub_dir = sub_dir[:-1]
            path = os.path.join(self.base_dir, sub_dir)
        else:
            path = self.base_dir
            
        os.makedirs(path, exist_ok=True)
            
        if filename:
            return os.path.join(path, filename)
        return path

    def record_audio(self):
        """Records audio from microphone and saves to WAV file."""
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.sample_format, 
            channels=self.channels, 
            rate=self.sampling_rate, 
            input=True, 
            frames_per_buffer=self.chunk
        )

        frames = []
        while self.recording: 
            data = stream.read(self.chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(self.audio_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(self.sample_format))
            wf.setframerate(self.sampling_rate)
            wf.writeframes(b''.join(frames))

def record(sub_dir="", filename=""):
    """
    Records video and audio simultaneously until 'q' is pressed.
    
    Args:
        sub_dir (str): Optional subdirectory to save the recording
        filename (str): Optional custom filename for the recording
    """
    try:
        video_class = RecordVideo()
        
        if sub_dir or filename:
            custom_path = video_class.update_path(sub_dir, filename)
            print(f"Using custom save path: {custom_path}")
        
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
        video_class.cap.release()
        video_class.video_writer.release()
        cv2.destroyAllWindows()
        
        print('Success')
        print(f"Recording finished. Video saved as {video_class.video_path} and audio saved as {video_class.audio_path}.")
        
    except Exception as e:
        print(f"Error: {e}")
