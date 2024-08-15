import time
import warnings
import logging
from typing import Generator, NoReturn
from contextlib import contextmanager

from RealtimeSTT import AudioToTextRecorder
import ai_model
import modules.tts_engine as tts_engine
import modules.logging_config as lc
import modules.json_handler as jh

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("langchain_core.callbacks.manager").setLevel(logging.ERROR)
logger = lc.configure_logger(__name__)
json_handler = jh.JsonHandler('config.json')

STT_MODEL = json_handler.get_setting('stt.model')

class ConversationManager:
    def __init__(self):
        self.asuka_ai = self._initialize_ai_model()
        self.tts = self._initialize_tts()
        self.recorder = self._initialize_recorder()

    def _initialize_ai_model(self):
        logger.info('Initializing AI Model...')
        return ai_model.AIModel(debug=False)

    def _initialize_tts(self):
        logger.info('Initializing TTS...')
        return tts_engine.TTS()

    def _initialize_recorder(self) -> AudioToTextRecorder:
        logger.info('Initializing recorder...')
        return AudioToTextRecorder(
            model=STT_MODEL,
            language='en',
            compute_type='int8',
            silero_use_onnx=True,
            spinner=False,
            silero_sensitivity=0.5
        )

    def process_input(self, input_text: str) -> None:
        if not input_text.strip():
            print("No input detected. Please try again.")
            return

        print(f'\nYOU: {input_text}\n', end="", flush=True)
        print('AI: ', end='', flush=True)
        self.tts.stream_inference(self.asuka_ai.yield_generate(input_text, True))
        print()

    @contextmanager
    def recording_session(self) -> Generator[None, None, None]:
        try:
            self.recorder.start()
            yield
        finally:
            input_text = self.recorder.stop().text()
            if input_text:
                self.process_input(input_text)

def main() -> NoReturn:
    logger.info('Initializing Conversation Manager...')
    manager = ConversationManager()

    while True:
        print('\nYou may speak now...')
        try:
            with manager.recording_session():
                time.sleep(0.1)  # Add a small delay to prevent rapid looping
        except KeyboardInterrupt:
            logger.info("Exiting the program...")
            break
        except Exception as e:
            logger.error(f"Error during conversation: {e}")
            time.sleep(1)

if __name__ == '__main__':
    main()