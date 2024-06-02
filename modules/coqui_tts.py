import os
import torch
from TTS.api import TTS
import modules.logging_config as lf
import modules.json_handler as jh

logger = lf.configure_logger(__name__)
json_handler = jh.JsonHandler('config.json')

# CoquiTTS settings
INPUT_FILE = json_handler.get_setting('tts.engines.coqui.input_file')
OUTPUT_FILE = json_handler.get_setting('tts.general_settings.output_file')
MODEL = json_handler.get_setting('tts.engines.coqui.model')

class CoquiTTS:
    def __init__(self, 
                 language: str = 'en', 
                 input_file: str = INPUT_FILE, 
                 output_file: str = OUTPUT_FILE,
                 model: str = MODEL):
        self.language = language
        self.input_file = input_file
        self.output_file = 'cache/' + output_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tts = TTS(model).to(self.device)
    
    def process_audio(self, input_text: str, speed: int = 3) -> None:
        try:
            if not os.path.exists(self.output_file):
                os.makedirs('cache', exist_ok=True)
            self.tts.tts_to_file(
                text=input_text,
                speaker_wav=self.input_file,
                language=self.language,
                file_path=self.output_file,
                speed=speed
            )
        except Exception as e:
            logger.error(f"An error occurred while processing audio: {e}")