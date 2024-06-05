import time
from RealtimeTTS import TextToAudioStream, CoquiEngine
import modules.logging_config as lf
import modules.json_handler as jh

logger = lf.configure_logger(__name__)

# Load configuration
json_handler = jh.JsonHandler('config.json')
REFERENCE_FILE = json_handler.get_setting('tts.reference_file')

class TTS:
    def __init__(self, 
                 reference_file: str = REFERENCE_FILE):
        self.engine = CoquiEngine(voice=reference_file,
                                  full_sentences=True)
        self.stream = TextToAudioStream(self.engine)

    def stream_inference(self, streamed_text: str) -> None:
        if self.stream.is_playing():
            self.stream.stop()
        self.stream.feed(streamed_text)
        self.stream.play_async()
        while self.stream.is_playing():
            time.sleep(0.1)  