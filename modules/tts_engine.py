import time
from RealtimeTTS import TextToAudioStream, CoquiEngine
import modules.logging_config as lc
import modules.json_handler as jh

logger = lc.configure_logger(__name__)

# Load configuration
json_handler = jh.JsonHandler('config.json')
REFERENCE_FILE = json_handler.get_setting('tts.reference_file')

class TTS:
    def __init__(self, reference_file: str = REFERENCE_FILE):
        self.engine = self._initialize_engine(reference_file)
        self.stream = TextToAudioStream(self.engine)

    def _initialize_engine(self, reference_file: str) -> CoquiEngine:
        try:
            return CoquiEngine(voice=reference_file, full_sentences=True)
        except Exception as e:
            logger.error(f"Failed to initialize CoquiEngine: {e}")
            raise

    def stream_inference(self, streamed_text: str) -> str:
        if self.stream.is_playing():
            self.stream.stop()
        self.stream.feed(streamed_text)
        self.stream.play_async()
        while self.stream.is_playing():
            time.sleep(0.1)