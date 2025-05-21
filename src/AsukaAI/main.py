# --- Global Warning Filters ---
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.getLogger("langchain_core.callbacks.manager").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Standard Library Imports ---
import time
import signal
import sys
from typing import NoReturn, Optional

# --- Project-Specific Imports ---
from audio.stt_engine import PushToTalkSTT
import ai_model
from audio import tts_engine
from utils import logging_config as lc
from utils import constants

# --- Setup Logger ---
logger = lc.configure_logger(__name__)

# --- Conversation Manager ---
class ConversationManager:
    def __init__(self):
        logger.info("Initializing Conversation Manager components...")
        try:
            # These initializations might raise RuntimeError if critical components fail
            self.stt: PushToTalkSTT = self._initialize_stt()
            self.asuka_ai: ai_model.AIModel = self._initialize_ai_model()
            self.tts: tts_engine.TTS = self._initialize_tts()
            logger.info("Conversation Manager components initialized successfully.")
        except (RuntimeError, ConnectionError) as e:
             logger.error(f"Fatal error during Conversation Manager initialization: {e}", exc_info=True)
             raise RuntimeError(f"Initialization failed: {e}") from e # Re-raise to be caught by Application
        except Exception as e:
             logger.error(f"Unexpected fatal error during Conversation Manager initialization: {e}", exc_info=True)
             raise RuntimeError(f"Unexpected initialization error: {e}") from e

    def _initialize_ai_model(self) -> ai_model.AIModel:
        logger.info('Initializing AI Model...')
        return ai_model.AIModel()

    def _initialize_tts(self) -> tts_engine.TTS:
        logger.info('Initializing TTS...')
        return tts_engine.TTS()

    def _initialize_stt(self) -> PushToTalkSTT:
        logger.info('Initializing STT engine (PushToTalkSTT)...')
        return PushToTalkSTT()

    def process_input(self, input_text: str) -> None:
        if not input_text or not input_text.strip():
            logger.warning("process_input called with empty or whitespace-only text.")
            return

        print(f'\nYOU: {input_text}\n', end="", flush=True)
        print('AI: ', end='', flush=True)

        full_response = ""
        ai_encountered_error = False
        try:
            for chunk in self.asuka_ai.stream(input_text):
                if chunk.startswith(constants.ERROR_MESSAGE_PREFIX): # Check AI stream errors
                    logger.error(f"Received error string from AI model stream: {chunk}")
                    error_message_display = " Sorry, I encountered an issue with that."
                    print(error_message_display, end='', flush=True)
                    ai_encountered_error = True
                    break
                full_response += chunk
                print(chunk, end='', flush=True)
            print() # Newline after AI response

            if full_response and not ai_encountered_error:
                self.tts.stream_inference(full_response)
            elif not full_response and not ai_encountered_error:
                 logger.warning("AI generated an empty response.")

        except Exception as e: # Catch errors from AI stream or TTS
            logger.error(f"Error during AI processing or TTS: {e}", exc_info=True)
            print("\n[Error processing your request]", flush=True)

    def close(self):
        logger.info("Closing Conversation Manager resources...")
        components_to_close = [
            ("STT engine", getattr(self, 'stt', None)),
            ("AI Model", getattr(self, 'asuka_ai', None)),
            ("TTS engine", getattr(self, 'tts', None)),
        ]
        for name, component in components_to_close:
            if component and hasattr(component, 'close') and callable(getattr(component, 'close')):
                try:
                    if constants.DEBUG_MODE: logger.debug(f"Closing {name}...")
                    component.close()
                    logger.info(f"{name} closed.")
                except Exception as e:
                    logger.error(f"Error closing {name}: {e}", exc_info=True)
            elif component:
                if constants.DEBUG_MODE: logger.debug(f"{name} exists but has no explicit 'close' method or is not callable.")
        logger.info("Conversation Manager resources closed.")

# --- Application Class for Encapsulation ---
class Application:
    def __init__(self):
        self.manager: Optional[ConversationManager] = None
        self._setup_signal_handlers()
        self._running = True

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        logger.warning(f"Signal {sig} received. Initiating graceful shutdown...")
        signal.signal(signal.SIGINT, signal.SIG_IGN) # Ignore further signals
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        self._running = False # Signal main loop to stop
        if self.manager: # Attempt to close manager immediately from handler as well
            logger.info("Signal handler calling manager.close().")
            self.manager.close()

    def run(self) -> NoReturn:
        exit_code = 0
        try:
            self.manager = ConversationManager() # Initialize components
            logger.info("Application ready. Starting conversation loop...")
            while self._running:
                try:
                    input_text = self.manager.stt.get_transcription(timeout=None)

                    if not self._running: break # Check flag again after blocking call

                    if input_text and input_text.startswith(constants.ERROR_MESSAGE_PREFIX):
                        logger.error(f"STT Error received: {input_text}")
                        error_msg_content = input_text[len(constants.ERROR_MESSAGE_PREFIX):].strip()
                        print(f"\n[Speech Recognition Error: {error_msg_content}]", flush=True)
                    elif input_text: # Valid transcription
                        self.manager.process_input(input_text)
                    else: # Silence or empty transcription
                        logger.info("No speech detected or transcription was empty after PTT release.")

                except RuntimeError as e: # Catch specific errors that might allow continuation
                    logger.error(f"Runtime error in conversation loop: {e}", exc_info=True)
                    print(f"\nA critical error occurred: {e}. Attempting to continue after a pause...", flush=True)
                    time.sleep(constants.RUNTIME_ERROR_WAIT_SECONDS)
                except Exception as e: # Catch other unexpected errors in the loop
                    logger.error(f"Unexpected error in conversation loop: {e}", exc_info=True)
                    print("\n[An unexpected error occurred in the loop. Trying to continue...]", flush=True)
                    time.sleep(constants.UNEXPECTED_LOOP_ERROR_WAIT_SECONDS)

        except SystemExit as se_obj:
            logger.info(f"SystemExit caught with code: {se_obj.code}")
            exit_code = se_obj.code if se_obj.code is not None else 0
        except RuntimeError as e:
            logger.critical(f"Fatal error during initialization: {e}", exc_info=constants.DEBUG_MODE)
            print(f"\nFATAL ERROR: Could not initialize application. {e}", file=sys.stderr, flush=True)
            exit_code = 1
        except Exception as e:
            logger.critical(f"Unexpected fatal error during setup: {e}", exc_info=True)
            print(f"\nUNEXPECTED FATAL ERROR: {e}", file=sys.stderr, flush=True)
            exit_code = 1
        finally:
            logger.info("Entering final cleanup phase (Application.run finally block)...")
            if self.manager:
                logger.info("Ensuring Conversation Manager resources are closed in finally block.")
                self.manager.close()
            logger.info(f"Program finished. Exiting with code {exit_code}.")
            sys.exit(exit_code)


def main_app_entry_point() -> NoReturn:
    logger.info(f"Application starting... (DEBUG_MODE: {'ON' if constants.DEBUG_MODE else 'OFF'})")
    app = Application()
    app.run()

if __name__ == '__main__':
    main_app_entry_point()