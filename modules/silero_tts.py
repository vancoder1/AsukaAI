import torch
import torchaudio
import modules.logging_config as lf

logger = lf.configure_logger(__name__)

class SileroTTS:
    def __init__(self, 
                 language: str = 'en', 
                 model_id: str = 'v3_en', 
                 speaker: str = 'en_107', 
                 sample_rate: int = 48000, 
                 output_file: str = r'cache\\ai_response_tts.mp3'):
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
                    format='mp3')
        except Exception as e:
            logger.error(f"An error occurred while processing audio: {e}")