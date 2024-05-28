import os
import keyboard
import numpy as np
import scipy.io.wavfile as wav
import threading
import time
from typing import List, Optional
from faster_whisper import WhisperModel
import modules.logging_config as lf

logger = lf.configure_logger(__name__)

FILE_LOCATION = 'cache\\dictate.wav'

class FasterWhisper:
    def __init__(self, 
                 file_location: str = FILE_LOCATION, 
                 model_size: str = 'distil-small.en', 
                 device: str = 'cpu'):
        self.file_location = file_location
        self.model_size = model_size
        self.device = device
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        self.model = WhisperModel(self.model_size, self.device, compute_type='int8')
        
    def transcribe(self) -> str:
        try:
            segments, _ = self.model.transcribe(self.file_location, vad_filter=True)
            text = ''.join(segment.text for segment in segments)
            return text
        except Exception as e:
            logger.error(f"An error occurred during transcription: {e}")
            return ""


class PushToTalkRecorder:
    def __init__(self, 
                 key: str = 'f8', 
                 sample_rate: int = 44100, 
                 channels: int = 1, 
                 output_file: str = FILE_LOCATION, 
                 min_recording_length: float = 0.2):
        self.key = key
        self.sample_rate = sample_rate
        self.channels = channels
        self.output_file = output_file
        self.min_recording_length = min_recording_length
        self.is_recording = False
        self.audio_data: List[np.ndarray] = []
        self.start_time: Optional[float] = None
        self.stream = None
        self.recording_done = threading.Event()
        self.sd = None

    def _lazy_import_sounddevice(self):
        if self.sd is None:
            import sounddevice as sd
            self.sd = sd
        
    def _audio_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def _start_stream(self):
        self._lazy_import_sounddevice()
        self.stream = self.sd.InputStream(samplerate=self.sample_rate, channels=self.channels, callback=self._audio_callback, blocksize=1024)
        self.stream.start()
        
    def _stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
    def _save_audio(self):
        if self.audio_data:
            audio_np = np.concatenate(self.audio_data, axis=0)
            wav.write(self.output_file, self.sample_rate, audio_np)
            print(f'Audio saved as {self.output_file}')
        else:
            logger.info('No audio data to save')
        
    def _on_key_event(self, event):
        if event.name == self.key:
            if event.event_type == 'down' and not self.is_recording:
                self.is_recording = True
                self.audio_data = []  # Reset audio data
                self.start_time = time.time()  # Record start time
                print('Recording started')
            elif event.event_type == 'up' and self.is_recording:
                self.is_recording = False
                duration = time.time() - self.start_time  # Calculate recording duration
                if duration >= self.min_recording_length:
                    self._save_audio()
                    print('Recording stopped', end='\n')
                    self.recording_done.set()
                    self.recording_done.clear()
                else:
                    print('Recording too short, not saved')
                
                
    def start(self):
        self._start_stream()  # Start the audio stream immediately
        keyboard.hook(self._on_key_event)
    
    def wait_for_recording(self):
        self.recording_done.wait()
