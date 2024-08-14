import time
import warnings
from typing import Generator
from contextlib import contextmanager

from RealtimeSTT import AudioToTextRecorder
import ai_model
import modules.tts_engine as tts_engine
import modules.logging_config as lc
import modules.json_handler as jh

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logger = lc.configure_logger(__name__)
json_handler = jh.JsonHandler('config.json')

STT_MODEL = json_handler.get_setting('stt.model')

class ConversationManager:
    def __init__(self):
        print('Initializing AI Model...')
        self.asuka_ai = ai_model.AIModel(debug=False)

        print('Initializing TTS...')
        self.tts = tts_engine.TTS()

        print('Initializing recorder...')
        self.recorder = self._initialize_recorder()

    def _initialize_recorder(self) -> AudioToTextRecorder:
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

def main():
    print('Initializing Conversation Manager...')
    manager = ConversationManager()

    while True:
        print('\nYou may speak now...')
        try:
            with manager.recording_session():
                time.sleep(0.1)  # Add a small delay to prevent rapid looping
        except KeyboardInterrupt:
            print("\nExiting the program...")
            break
        except Exception as e:
            logger.error(f"Error during conversation: {e}")
            time.sleep(1)

if __name__ == '__main__':
    main()