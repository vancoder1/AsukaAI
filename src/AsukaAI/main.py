import time
import warnings
import logging
from typing import NoReturn
import signal # For graceful shutdown
import sys # For sys.exit

# Imports for modules within the AsukaAI package
from audio.stt_engine import PushToTalkSTT
import ai_model
from audio import tts_engine
from utils import logging_config as lc
from utils import json_handler as jh

# --- Filter Warnings ---
# Filter common warnings to keep output clean
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Filter specific noisy loggers if needed
logging.getLogger("langchain_core.callbacks.manager").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING) # Example: Quieten httpx logs

# --- Setup Logger ---
logger = lc.configure_logger(__name__)

# --- Conversation Manager ---
class ConversationManager:
    """
    Manages the STT, AI Model, and TTS components for a conversation flow.
    """
    def __init__(self):
        logger.info("Initializing Conversation Manager components...")
        # Initialize components, handling potential errors during setup
        try:
            # These methods should return initialized instances
            self.stt = self._initialize_stt()
            self.asuka_ai = self._initialize_ai_model()
            self.tts = self._initialize_tts()
            logger.info("Conversation Manager components initialized successfully.")
        except (RuntimeError, ConnectionError) as e:
             logger.error(f"Fatal error during Conversation Manager initialization: {e}", exc_info=False) # Log concise error
             # Re-raise to be caught by the main loop's exception handler
             raise RuntimeError(f"Initialization failed: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected fatal error during Conversation Manager initialization: {e}", exc_info=True)
             raise RuntimeError(f"Unexpected initialization error: {e}") from e


    def _initialize_ai_model(self) -> ai_model.AIModel:
        logger.info('Initializing AI Model...')
        # Pass debug flag if needed, e.g., based on environment variable or config
        return ai_model.AIModel(debug=False)

    def _initialize_tts(self) -> tts_engine.TTS:
        logger.info('Initializing TTS...')
        return tts_engine.TTS()

    def _initialize_stt(self) -> PushToTalkSTT:
        logger.info('Initializing STT engine (PushToTalkSTT)...')
        # This now correctly initializes the refactored STT class
        return PushToTalkSTT()

    def process_input(self, input_text: str) -> None:
        """Processes transcribed text, gets AI response, and speaks it."""
        if not input_text or not input_text.strip():
            logger.warning("process_input called with empty or whitespace-only text.")
            # Optionally provide feedback to the user via TTS or print
            # print("No valid input detected. Please try again.")
            return

        print(f'\nYOU: {input_text}\n', end="", flush=True)
        print('AI: ', end='', flush=True)

        full_response = ""
        try:
            # Use the stream method directly
            for chunk in self.asuka_ai.stream(input_text):
                if "[ERROR:" in chunk: # Handle potential errors from AI stream
                    logger.error(f"Received error chunk from AI model: {chunk}")
                    # Decide how to handle AI errors (e.g., stop, say error message)
                    full_response += " Sorry, I encountered an error."
                    print(" Sorry, I encountered an error.", end='', flush=True)
                    break # Stop processing this response
                full_response += chunk
                print(chunk, end='', flush=True)
            print() # Newline after AI response

            if full_response and not full_response.startswith(" Sorry, I encountered an error."):
                self.tts.stream_inference(full_response)
            elif not full_response:
                 logger.warning("AI generated an empty response.")
                 # Optionally say something like "I didn't generate a response."

        except Exception as e:
            logger.error(f"Error during AI processing or TTS: {e}", exc_info=True)
            # Optionally inform user via print/TTS about the error
            print("\n[Error processing request]", flush=True)


    def close(self):
        """Clean up resources for all managed components."""
        logger.info("Closing Conversation Manager resources...")
        # Use hasattr for safe cleanup, in case initialization failed partially
        if hasattr(self, 'stt') and self.stt and hasattr(self.stt, 'close'):
            try:
                logger.debug("Closing STT engine...")
                self.stt.close()
                logger.info("STT engine closed.")
            except Exception as e:
                logger.error(f"Error closing STT engine: {e}", exc_info=True)

        if hasattr(self, 'tts') and self.tts and hasattr(self.tts, 'close'):
            try:
                logger.debug("Closing TTS engine...")
                self.tts.close()
                logger.info("TTS engine closed.")
            except Exception as e:
                logger.error(f"Error closing TTS engine: {e}", exc_info=True)

        # Add cleanup for AI model if needed (e.g., closing files, connections)
        if hasattr(self, 'asuka_ai') and self.asuka_ai:
             # if hasattr(self.asuka_ai, 'close'): self.asuka_ai.close()
             logger.debug("AI model resources implicitly managed (no explicit close needed).")

        logger.info("Conversation Manager resources closed.")

# --- Signal Handler for Graceful Shutdown ---
# Define manager globally for access within the handler
manager: ConversationManager | None = None

def signal_handler(sig, frame):
    """Handles SIGINT (Ctrl+C) and SIGTERM for graceful shutdown."""
    logger.warning(f"Signal {sig} received. Initiating graceful shutdown...")
    # Prevent recursive calls if shutdown takes time or signal is repeated
    signal.signal(signal.SIGINT, signal.SIG_IGN) # Ignore further SIGINT
    signal.signal(signal.SIGTERM, signal.SIG_IGN) # Ignore further SIGTERM

    if manager:
        manager.close() # Call the cleanup method
    logger.info("Shutdown complete.")
    sys.exit(0) # Exit cleanly

# --- Main Application Logic ---
def main() -> NoReturn:
    """Main application entry point."""
    global manager # Allow modification of the global manager variable

    # Register signal handlers *before* initializing potentially blocking resources
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Handle termination signals (e.g., kill)

    try:
        manager = ConversationManager() # Initialization happens here

        # Main conversation loop
        while True:
            try:
                # Wait for PTT press/release and get transcription
                input_text = manager.stt.get_transcription(timeout=None) # Wait for result

                # Process the input if it's valid and not an STT error
                if input_text and not input_text.startswith("[ERROR"):
                    manager.process_input(input_text)
                elif input_text.startswith("[ERROR"):
                    logger.error(f"STT Error received: {input_text}")
                    # Optionally inform the user via TTS or print
                    error_msg = input_text.split(':', 1)[-1].strip()
                    print(f"\n[STT Error: {error_msg}]")
                    # Decide whether to continue or break on STT error
                    # continue # Continue to next PTT cycle
                else:
                    # Handle cases of silence or empty transcription after PTT release
                    logger.info("No speech detected or transcription was empty.")
                    # Optionally provide feedback: print("Didn't catch that.")

            except RuntimeError as e:
                 # Catch specific runtime errors from STT/TTS/AI if they signal critical issues
                 logger.error(f"Runtime error in conversation loop: {e}", exc_info=True)
                 print(f"\nA critical error occurred: {e}. Attempting to continue...")
                 time.sleep(2) # Pause before potentially retrying
            except Exception as e:
                 # Catch unexpected errors during the STT wait or input processing
                 logger.error(f"Unexpected error in conversation loop: {e}", exc_info=True)
                 print("\n[An unexpected error occurred in the loop. Trying to continue...]")
                 # Add a small delay to prevent spamming logs if errors repeat quickly
                 time.sleep(1)

    # Removed KeyboardInterrupt handler - SIGINT is caught by signal_handler
    except SystemExit as se:
         logger.info(f"SystemExit caught: {se}") # Catch exits from signal handler or init failures
    except RuntimeError as e:
         # Catch initialization errors from ConversationManager
         logger.critical(f"Fatal error during initialization: {e}", exc_info=False) # Details logged inside manager
         print(f"\nFATAL ERROR: Could not initialize application. {e}", file=sys.stderr)
         sys.exit(1) # Exit with error code
    except Exception as e:
         # Catch any other unexpected errors during setup before the main loop
         logger.critical(f"Unexpected fatal error during setup: {e}", exc_info=True)
         print(f"\nUNEXPECTED FATAL ERROR: {e}", file=sys.stderr)
         sys.exit(1) # Exit with error code
    finally:
        # Ensure resources are closed even if errors occur before signal handling
        # or if signal handling fails. This might run even after signal_handler's close.
        logger.info("Entering final cleanup phase (main finally block)...")
        # Check if manager exists and hasn't been cleaned yet (might be redundant if signal handler always works)
        if manager:
            # It's generally safe to call close multiple times if the close methods are idempotent
            manager.close()
        logger.info("Program finished.")
        # Explicit exit call might be needed depending on lingering non-daemon threads
        # sys.exit(0) # Exiting via signal handler or error code above is preferred

if __name__ == '__main__':
    logger.info("Application starting...")
    main()
