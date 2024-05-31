import torch
from TTS.api import TTS
import modules.logging_config as lf

logger = lf.configure_logger(__name__)

class Coqui_TTS:
    def __init__(self, 
                 language: str = 'en', 
                 input_file: str = r'datasets\\xtts_voice\\input.wav', 
                 output_file: str = r'cache\\ai_response_tts.wav'):
        self.language = language
        self.input_file = input_file
        self.output_file = output_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to(self.device)
    
    def process_audio(self, input_text: str, speed: int = 3):
        try:
            self.tts.tts_to_file(text=input_text,
                                 speaker_wav=self.input_file,
                                 language=self.language,
                                 file_path=self.output_file,
                                 speed=speed)
            print(f"Audio processed successfully and saved to {self.output_file}")
        except Exception as e:
            logger.error(f"An error occurred while processing audio: {e}")