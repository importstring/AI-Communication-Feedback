import cv2
import pyaudio
import wave
import threading

class RecordVideo:
    def __init__(self):
        self.video_filename = 'output.mp4'
        self.frame_rate = 60

        self.audio_filename = 'output.wav'  
        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.sampling_rate = 44100

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Unable to access the camera.")
        
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Fixed codec to 'mp4v'
        self.video_writer = cv2.VideoWriter(self.video_filename, self.fourcc, self.frame_rate, (640, 480)) 

        self.base_dir = '../'

        self.recording = False
        
        self.video_filename = self.update_path(self.base_dir + '/recordings', self.video_filename)
        self.audio_filename = self.update_path(self.base_dir + '/recordings', self.audio_filename)

    def update_path(self, sub_dir='', filename=''):
        output = self.base_dir
        output += sub_dir if sub_dir != "" else ""
        output += '/' + filename if filename != '' else ""
        return output

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

        while self.recording: 
            data = stream.read(self.chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(self.audio_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(self.sample_format))
            wf.setframerate(self.sampling_rate)
            wf.writeframes(b''.join(frames))

def record(sub_dir="", filename=""):
    video_class = RecordVideo()
    video_class.save_path = video_class.update_path(sub_dir, filename)
    
    video_class.recording = True
    audio_thread = threading.Thread(target=video_class.record_audio)  # Fixed threading target
    audio_thread.start()

    print(r"Press 'q' to stop recording.")
    
    while video_class.recording:  # Fixed variable name
        ret, frame = video_class.cap.read()  # Fixed variable name
        if not ret:
            break

        video_class.video_writer.write(frame)  # Fixed variable name
        cv2.imshow('Recording', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_class.recording = False
    
    audio_thread.join()
    video_class.cap.release()  # Fixed variable name
    video_class.video_writer.release()  # Fixed variable name
    cv2.destroyAllWindows()
    
    print('Success')
    print(f"Recording finished. Video saved as {video_class.video_filename} and audio saved as {video_class.audio_filename}.")

# Testing the code for syntax and runtime errors
try:
    record()
except Exception as e:
    print(f"Error: {e}")
