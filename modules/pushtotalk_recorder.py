import os
import time
import keyboard
import pyaudio
import wave
import threading
import modules.logging_config as lf
import modules.json_handler as jh

logger = lf.configure_logger(__name__)
json_handler = jh.JsonHandler('config.json')

KEY = json_handler.get_setting('recorder.key')
SAMPLE_RATE = json_handler.get_setting('recorder.sample_rate')
CHANNELS = json_handler.get_setting('recorder.channels')
CHUNK = json_handler.get_setting('recorder.chunk')
FILE_LOCATION = json_handler.get_setting('recorder.output_file')

class PushToTalkRecorder:
    def __init__(self, 
                 key: str = KEY, 
                 sample_rate: int = SAMPLE_RATE, 
                 channels: int = CHANNELS, 
                 chunk: int = CHUNK,
                 output_file: str = FILE_LOCATION, 
                 min_recording_length: float = 0.2):
        self.key = key
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.output_file = 'cache/' + output_file
        self.min_recording_length = min_recording_length
        self.is_recording = False
        self.audio_data = []
        self.start_time = None
        self.recording_done = threading.Event()
        self.p = pyaudio.PyAudio()
        self.stream = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.audio_data.append(in_data)
        return None, pyaudio.paContinue

    def _start_stream(self) -> None:
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._audio_callback
        )

    def _stop_stream(self) -> None:
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

    def _save_audio(self) -> None:
        if not os.path.exists(self.output_file):
            os.makedirs('cache', exist_ok=True)
        
        if self.audio_data:
            with wave.open(self.output_file, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.audio_data))
        else:
            logger.warning('No audio data to save')

    def _on_key_event(self, event: keyboard.KeyboardEvent) -> None:
        if event.name == self.key:
            if event.event_type == 'down' and not self.is_recording:
                self.is_recording = True
                self.audio_data = []
                self.start_time = time.time()
                logger.info('Recording...')
            elif event.event_type == 'up' and self.is_recording:
                self.is_recording = False
                duration = time.time() - self.start_time
                if duration >= self.min_recording_length:
                    self._save_audio()
                    logger.info('Recording stopped')
                    self.recording_done.set()
                else:
                    logger.warning('Recording too short, not saved')
                self.recording_done.clear()

    def start(self) -> None:
        self._start_stream()
        keyboard.hook(self._on_key_event)

    def wait_for_recording(self) -> None:
        self.recording_done.wait()

    def __del__(self):
        self._stop_stream()
        self.p.terminate()