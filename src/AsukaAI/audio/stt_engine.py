import sounddevice as sd
import numpy as np
import keyboard
import threading
import queue
import time
import logging
from faster_whisper import WhisperModel
import sys # Keep for potential error exit

# --- Simplified Utility Import ---
# Attempt to import custom utils, fall back gracefully.
try:
    # Try absolute import first (adjust 'AsukaAI' if your project root differs)
    from AsukaAI.utils import logging_config as lc
    from AsukaAI.utils import json_handler as jh
    logger = lc.configure_logger(__name__)
    CONFIG_FILE = 'config/config.json'
    json_handler = jh.JsonHandler(CONFIG_FILE)
    logger.info("Successfully imported custom utility modules.")
except ImportError:
    # Fallback to relative import
    try:
        import utils.logging_config as lc
        import utils.json_handler as jh
        logger = lc.configure_logger(__name__)
        CONFIG_FILE = 'config/config.json'
        json_handler = jh.JsonHandler(CONFIG_FILE)
        logger.info("Successfully imported custom utility modules (relative path).")
    except ImportError:
        # Fallback to basic logging and default config path if utils are missing
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        logger.warning("Failed to import custom utility modules. Using basic logging and default settings.")
        CONFIG_FILE = 'config/config.json' # Assume default location

        # Dummy JsonHandler to avoid errors later if file is missing
        class DummyJsonHandler:
            def __init__(self, path):
                self.path = path
            def get_setting(self, key, default):
                # Cannot read from file, always return default
                return default
        json_handler = DummyJsonHandler(CONFIG_FILE)


# --- Configuration Loading ---
# Define defaults in one place
DEFAULT_SETTINGS = {
    'stt.model_size': 'base.en',
    'stt.device': 'cpu',
    'stt.compute_type': 'int8',
    'stt.ptt_key': 'x',
    'stt.language': 'en',
    'stt.beam_size': 5, # Added beam_size to config
}

# Load settings using a loop for simplicity
config = {}
for key, default_value in DEFAULT_SETTINGS.items():
    try:
        # Use the imported or dummy json_handler
        config[key] = json_handler.get_setting(key, default_value)
    except Exception as e:
        # Catch potential errors within the JsonHandler itself
        logger.error(f"Error reading setting '{key}' from {CONFIG_FILE}: {e}. Using default: {default_value}")
        config[key] = default_value

# Assign to constants/variables (using .get ensures keys exist)
STT_MODEL_SIZE = config.get('stt.model_size')
STT_DEVICE = config.get('stt.device')
STT_COMPUTE_TYPE = config.get('stt.compute_type')
PTT_KEY = config.get('stt.ptt_key')
LANGUAGE_CODE = config.get('stt.language')
BEAM_SIZE = config.get('stt.beam_size')

logger.info("--- STT Configuration ---")
logger.info(f"  Model:         {STT_MODEL_SIZE}")
logger.info(f"  Device:        {STT_DEVICE}")
logger.info(f"  Compute Type:  {STT_COMPUTE_TYPE}")
logger.info(f"  PTT Key:       {PTT_KEY}")
logger.info(f"  Language:      {LANGUAGE_CODE}")
logger.info(f"  Beam Size:     {BEAM_SIZE}")
logger.info("-------------------------")


# --- Audio Stream Constants ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.int16
# Block size can be adjusted, but 100ms (1600 frames) is reasonable
BLOCK_DURATION_MS = 100
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)


class PushToTalkSTT:
    """
    Handles speech-to-text transcription using faster-whisper, triggered
    by a push-to-talk (PTT) key. Records audio while the key is held
    and transcribes the entire recording upon release.
    """
    def __init__(self):
        logger.info("Initializing PushToTalkSTT...")

        self.ptt_key = PTT_KEY
        self.language_code = LANGUAGE_CODE
        self.beam_size = BEAM_SIZE

        self._model = self._load_model()

        self._audio_buffer = []       # Holds audio chunks during PTT press
        self._result_queue = queue.Queue(maxsize=1) # Passes final transcription
        self._lock = threading.Lock() # Protects _ptt_pressed and _audio_buffer
        self._ptt_pressed = False     # Tracks the PTT key state

        # Stream-related state
        self._stream_thread = None
        self._stream_stop_event = threading.Event()
        self._stream_active = threading.Event() # Signals when stream is ready

        # Transcription state
        self._transcription_thread = None # Thread for the transcription task

        self._setup_ptt_listener()
        self._start_stream_thread()

    def _load_model(self):
        """Loads the faster-whisper model."""
        try:
            logger.info(f"Loading FasterWhisper model: {STT_MODEL_SIZE} (Device: {STT_DEVICE}, Compute: {STT_COMPUTE_TYPE})")
            model = WhisperModel(STT_MODEL_SIZE, device=STT_DEVICE, compute_type=STT_COMPUTE_TYPE)
            logger.info("FasterWhisper model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Fatal: Failed to load FasterWhisper model '{STT_MODEL_SIZE}'. Error: {e}", exc_info=True)
            # Option 1: Raise error to stop application
            raise RuntimeError("STT Model loading failed. Check logs.")
            # Option 2: Return None and handle it later (less safe)
            # return None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        """Callback function for the audio stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        # Use lock to safely check flag and append to buffer
        with self._lock:
            if self._ptt_pressed:
                # Append a copy to avoid issues if indata buffer is reused
                self._audio_buffer.append(indata.copy())

    def _manage_audio_stream(self):
        """Manages the audio input stream in a separate thread."""
        stream = None
        while not self._stream_stop_event.is_set():
            try:
                logger.info("Attempting to start audio stream...")
                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype=DTYPE,
                    blocksize=BLOCK_SIZE,
                    callback=self._audio_callback
                )
                with stream:
                    self._stream_active.set() # Signal stream is active
                    logger.info(f"Audio stream started. Ready for PTT ('{self.ptt_key}').")
                    # Wait until stop event is set or stream becomes inactive
                    while stream.active and not self._stream_stop_event.is_set():
                        time.sleep(0.1)
                    logger.info("Exiting audio stream context...")

            except sd.PortAudioError as pae:
                logger.error(f"PortAudio error: {pae}. Check audio devices.")
                self._stream_stop_event.set() # Stop trying on severe error
                break # Exit loop
            except Exception as e:
                logger.error(f"Unexpected error in audio stream thread: {e}", exc_info=True)
                # Avoid busy-looping on repeated errors
                if self._stream_stop_event.wait(timeout=5): # Wait up to 5s if stop is requested
                    break # Exit loop if stop was requested during sleep
                logger.info("Attempting to restart audio stream after error...")
            finally:
                self._stream_active.clear() # Signal stream is inactive
                if stream:
                    # Ensure stream resources are released if an error occurred within 'with'
                    try:
                        if not stream.closed:
                            stream.stop()
                            stream.close()
                            logger.debug("Stream stopped and closed in finally block.")
                    except Exception as e:
                        logger.warning(f"Error closing stream in finally block: {e}")
                stream = None # Reset stream variable

            if not self._stream_stop_event.is_set():
                 logger.warning("Audio stream stopped. Retrying in 5 seconds...")
                 time.sleep(5) # Wait before retrying

        self._stream_active.clear()
        logger.info("Audio stream management thread finished.")

    def _start_stream_thread(self):
        """Starts the audio stream management thread if not already running."""
        if self._stream_thread is None or not self._stream_thread.is_alive():
            self._stream_stop_event.clear()
            self._stream_active.clear()
            self._stream_thread = threading.Thread(target=self._manage_audio_stream, daemon=True)
            self._stream_thread.start()
            logger.info("Audio stream management thread started.")
        else:
            logger.info("Audio stream management thread already running.")

    def _on_ptt_press(self, event=None): # Accept optional event arg
        """Handles the PTT key press event."""
        with self._lock:
            if not self._ptt_pressed:
                logger.info(f"'{self.ptt_key}' pressed, recording started...")
                self._ptt_pressed = True
                self._audio_buffer = [] # Clear buffer for new recording
                # Clear previous result from queue if any
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass

    def _on_ptt_release(self, event=None): # Accept optional event arg
        """Handles the PTT key release event."""
        audio_to_process = None
        with self._lock:
            if self._ptt_pressed:
                logger.info(f"'{self.ptt_key}' released, preparing for transcription...")
                self._ptt_pressed = False
                # Capture the buffer content *before* clearing
                if self._audio_buffer:
                    audio_to_process = np.concatenate(self._audio_buffer, axis=0)
                self._audio_buffer = [] # Clear buffer immediately after copy/concat

        # Perform transcription outside the lock and potentially in a new thread
        if audio_to_process is not None and audio_to_process.size > 0:
             # Don't block the keyboard listener - run transcription in a separate thread
             if self._transcription_thread and self._transcription_thread.is_alive():
                 logger.warning("Previous transcription is still running. Discarding new request.")
                 self._put_result("[WARN: Transcription overlapped]")
             else:
                logger.debug(f"Starting transcription for {len(audio_to_process)/SAMPLE_RATE:.2f}s audio...")
                self._transcription_thread = threading.Thread(
                    target=self._perform_transcription,
                    args=(audio_to_process,),
                    daemon=True # Allow program exit even if this thread hangs
                )
                self._transcription_thread.start()
        else:
             logger.info("PTT released, but no audio recorded.")
             self._put_result("") # Put empty result if no audio

    def _perform_transcription(self, audio_np: np.ndarray):
        """Transcribes the given audio data using the loaded model."""
        transcription = "[ERROR: Transcription failed]" # Default error message
        try:
            # 1. Convert to float32 required by Whisper
            # Normalization factor for int16 to float32 range [-1.0, 1.0]
            audio_float32 = audio_np.astype(np.float32) / 32768.0

            # --- FIX: Ensure audio is a 1D array ---
            # faster-whisper expects a 1D array (N,), but our concatenation
            # likely produced (N, 1). We flatten or squeeze it.
            if audio_float32.ndim > 1:
                logger.debug(f"Original audio shape: {audio_float32.shape}. Reshaping to 1D.")
                # Use squeeze() to remove single-dimensional entries. Handles (N, 1) -> (N,)
                # Use flatten() to always create a 1D array, also safe here.
                audio_float32 = audio_float32.squeeze()
                # Alternative: audio_float32 = audio_float32.flatten()
                logger.debug(f"Reshaped audio shape: {audio_float32.shape}")

            # Add a check for empty audio after potential squeeze/flatten
            if audio_float32.size == 0:
                logger.warning("Audio data is empty after processing. Skipping transcription.")
                transcription = "" # Return empty string for no audio
                self._put_result(transcription)
                return # Exit the function early

            # 2. Perform transcription
            logger.debug(f"Transcribing audio of size {audio_float32.size} samples ({audio_float32.size / SAMPLE_RATE:.2f}s)")
            segments, info = self._model.transcribe(
                audio_float32, # Pass the corrected 1D array
                language=self.language_code if self.language_code else None, # Pass None for auto-detect
                beam_size=self.beam_size,
                # condition_on_previous_text=False # Not needed for single transcription
                # vad_filter=True # Optional: Use VAD for better segmentation
            )

            # 3. Concatenate segments
            transcription = "".join(segment.text for segment in segments).strip()

            logger.info(f"Detected language: {info.language} (p={info.language_probability:.2f})")
            logger.debug(f"Raw transcription: '{transcription}'")

        except Exception as e:
            # Log the specific error during transcription
            logger.error(f"Error during transcription: {e}", exc_info=True)
            # Keep the default error message "[ERROR: Transcription failed]"

        finally:
            # 4. Put the result (or error) into the queue
            self._put_result(transcription)
            logger.debug("Transcription thread finished.")

    def _put_result(self, text: str):
        """Safely puts the final transcription result into the queue."""
        try:
            # Clear queue first to ensure only the latest result is present
            while not self._result_queue.empty():
                try: self._result_queue.get_nowait()
                except queue.Empty: break
            # Put the new result
            self._result_queue.put_nowait(text)
            logger.debug(f"Put result in queue: '{text[:50]}...'")
        except queue.Full:
            # Should not happen with maxsize=1 after clearing, but log defensively
            logger.error("Result queue is full even after clearing. Race condition?")


    def _setup_ptt_listener(self):
        """Registers the PTT key listeners using the keyboard library."""
        try:
            # Use press/release_key for specific key handling
            keyboard.on_press_key(self.ptt_key, self._on_ptt_press, suppress=False)
            keyboard.on_release_key(self.ptt_key, self._on_ptt_release, suppress=False)
            # Consider using keyboard.add_hotkey for combined press/release logic if needed
            logger.info(f"PTT hotkeys ('{self.ptt_key}') registered successfully.")
        except ImportError:
            logger.error("Fatal: The 'keyboard' library is not installed or cannot be accessed.", exc_info=True)
            logger.error("Try running with sudo (Linux/macOS) or as Administrator (Windows) if permissions are denied.")
            raise RuntimeError("Missing 'keyboard' dependency or insufficient permissions.")
        except Exception as e:
            logger.error(f"Fatal: Failed to register PTT hotkey '{self.ptt_key}'. Error: {e}", exc_info=True)
            raise RuntimeError("Hotkey registration failed. Check logs.")

    def get_transcription(self, timeout: float = 30.0) -> str:
        """
        Waits for a PTT press-and-release cycle and returns the transcription.

        Blocks until a transcription is available after PTT release or timeout.

        Args:
            timeout (float): Maximum time in seconds to wait for a result.
                             Defaults to 30.0. Use None for no timeout.

        Returns:
            str: The transcribed text, or an error message/empty string on timeout/error.
        """
        # 1. Ensure the stream is running and ready
        if not self._stream_active.wait(timeout=5.0):
            logger.error("Audio stream failed to become active. Cannot get transcription.")
            return "[ERROR: Audio stream inactive]"

        # 2. Wait for a result to be placed in the queue by the release handler/transcription thread
        logger.info(f"Ready. Press and hold '{self.ptt_key}', release to transcribe.")
        try:
            # Block until the transcription thread puts the result
            result = self._result_queue.get(timeout=timeout)
            logger.info("Transcription received from queue.")
            return result
        except queue.Empty:
            logger.warning(f"Timeout ({timeout}s) waiting for transcription result.")
            return "[ERROR: Timeout waiting for transcription]"

    def close(self):
        """Cleans up resources: stops threads, closes streams, unregisters hotkeys."""
        logger.info("Cleaning up STT engine...")

        # 1. Unregister hotkeys
        try:
            # Use unhook_all for simplicity, assumes this is the only keyboard use
            keyboard.unhook_all()
            logger.info("Keyboard hooks removed.")
        except Exception as e:
            # Log warning instead of error, cleanup should continue
            logger.warning(f"Error removing keyboard hooks: {e}")

        # 2. Signal the stream management thread to stop
        self._stream_stop_event.set()

        # 3. Wait for the stream management thread to finish
        if self._stream_thread and self._stream_thread.is_alive():
            logger.info("Waiting for audio stream thread to join...")
            self._stream_thread.join(timeout=5.0) # Wait up to 5 seconds
            if self._stream_thread.is_alive():
                logger.warning("Audio stream thread did not stop gracefully.")
        else:
            logger.info("Audio stream thread was not running or already joined.")

        # 4. Wait briefly for any ongoing transcription thread (optional but good practice)
        if self._transcription_thread and self._transcription_thread.is_alive():
             logger.info("Waiting briefly for potential ongoing transcription...")
             self._transcription_thread.join(timeout=5.0) # Adjust timeout as needed
             if self._transcription_thread.is_alive():
                 logger.warning("Transcription thread still active during cleanup.")

        # 5. Clear internal buffer and queue as a final step
        with self._lock:
            self._audio_buffer = []
        while not self._result_queue.empty():
            try: self._result_queue.get_nowait()
            except queue.Empty: break

        logger.info("STT engine cleanup complete.")
