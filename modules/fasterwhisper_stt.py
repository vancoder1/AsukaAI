import os
from faster_whisper import WhisperModel
import modules.logging_config as lf
import modules.json_handler as jh

logger = lf.configure_logger(__name__)
json_handler = jh.JsonHandler('config.json')

# STT settings
FILE_LOCATION = json_handler.get_setting('recorder.output_file')
MODEL_SIZE = json_handler.get_setting('stt.model')

class FasterWhisper:
    def __init__(self, 
                 file_location: str = FILE_LOCATION, 
                 model_size: str = MODEL_SIZE, 
                 device: str = 'cpu'):
        self.file_location = 'cache/' + file_location
        self.model_size = model_size
        self.device = device
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        self.model = WhisperModel(self.model_size, self.device, compute_type='int8')
        
    def transcribe(self) -> str:
        try:
            if not os.path.exists(self.file_location):
                os.makedirs('cache', exist_ok=True)
            segments, _ = self.model.transcribe(self.file_location, vad_filter=True)
            text = ''.join(segment.text for segment in segments)
            return text
        except Exception as e:
            logger.error(f"An error occurred during transcription: {e}")
            return ""


