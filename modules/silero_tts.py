import torch
import torchaudio
import modules.logging_config as lf
import modules.json_handler as jh

logger = lf.configure_logger(__name__)
json_handler = jh.JsonHandler('config.json')

# SileroTTS settings
OUTPUT_FILE = json_handler.get_setting('tts.general_settings.output_file')
SAMPLE_RATE = json_handler.get_setting('tts.general_settings.sample_rate')
LANGUAGE = json_handler.get_setting('tts.engines.silero.language')
MODEL_ID = json_handler.get_setting('tts.engines.silero.model_id')
SPEAKER = json_handler.get_setting('tts.engines.silero.speaker')

class SileroTTS:
    def __init__(self, 
                 language: str = LANGUAGE, 
                 model_id: str = MODEL_ID, 
                 speaker: str = SPEAKER, 
                 sample_rate: int = SAMPLE_RATE, 
                 output_file: str = OUTPUT_FILE):
        self.language = language
        self.model_id = model_id
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.output_file = output_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                    model='silero_tts',
                                    language=self.language,
                                    speaker=self.model_id).to(self.device)
    
    def process_audio(self, input_text: str):
        try:
            audio = self.model.apply_tts(text=input_text,
                            speaker=self.speaker,
                            sample_rate=self.sample_rate)
            torchaudio.save(self.output_file,
                    audio.unsqueeze(0),
                    sample_rate=self.sample_rate,
                    format='wav')
        except Exception as e:
            logger.error(f"An error occurred while processing audio: {e}")