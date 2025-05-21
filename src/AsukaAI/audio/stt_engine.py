# stt_engine.py
import sounddevice as sd
import numpy as np
import keyboard # Ensure this import is robust or handled
import threading
import queue
import time
from typing import Optional # For type hinting

from faster_whisper import WhisperModel

from utils import logging_config as lc
from utils import json_handler as jh
from utils import constants # Import constants
from utils.config_loader import load_settings_from_json

logger = lc.configure_logger(__name__) # Logger level will respect DEBUG_MODE
CONFIG_FILE = constants.CONFIG_FILE_PATH

# --- Configuration Loading ---
STT_CONFIG_DEFS = {
    'STT_MODEL_SIZE': ('stt.model_size', 'base.en'),
    'STT_DEVICE': ('stt.device', 'cpu'),
    'STT_COMPUTE_TYPE': ('stt.compute_type', 'int8'),
    'PTT_KEY': ('stt.ptt_key', 'x'),
    'LANGUAGE_CODE': ('stt.language', 'en'), # 'en' or None for auto-detect
    'BEAM_SIZE': ('stt.beam_size', 5),
}

stt_json_handler = jh.JsonHandler(CONFIG_FILE)
_config_values = load_settings_from_json(logger, stt_json_handler, STT_CONFIG_DEFS, "STT")

STT_MODEL_SIZE = _config_values['STT_MODEL_SIZE']
STT_DEVICE = _config_values['STT_DEVICE']
STT_COMPUTE_TYPE = _config_values['STT_COMPUTE_TYPE']
PTT_KEY = _config_values['PTT_KEY']
LANGUAGE_CODE = _config_values['LANGUAGE_CODE']
BEAM_SIZE = _config_values['BEAM_SIZE']

if constants.DEBUG_MODE:
    logger.debug("--- STT Effective Configuration ---")
    logger.debug(f"  Model:         {STT_MODEL_SIZE}")
    logger.debug(f"  Device:        {STT_DEVICE}")
    logger.debug(f"  Compute Type:  {STT_COMPUTE_TYPE}")
    logger.debug(f"  PTT Key:       {PTT_KEY}")
    logger.debug(f"  Language:      {LANGUAGE_CODE if LANGUAGE_CODE else 'auto'}")
    logger.debug(f"  Beam Size:     {BEAM_SIZE}")
    logger.debug("--------------------------------")


# --- Audio Stream Constants ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.int16
BLOCK_DURATION_MS = 100 # Stream buffer granularity
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)


class PushToTalkSTT:
    def __init__(self):
        logger.info("Initializing PushToTalkSTT...")

        self.ptt_key = PTT_KEY
        self.language_code = LANGUAGE_CODE if LANGUAGE_CODE and LANGUAGE_CODE.strip() else None # Ensure None if empty string
        self.beam_size = BEAM_SIZE

        self._model: Optional[WhisperModel] = None # Initialize as optional
        try:
            self._model = self._load_model()
        except RuntimeError as e:
            logger.critical(f"Failed to load Whisper model during STT init: {e}", exc_info=True)
            raise # Re-raise to signal critical failure

        self._audio_buffer = []
        self._result_queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._ptt_pressed = False

        self._stream_thread: Optional[threading.Thread] = None
        self._stream_stop_event = threading.Event()
        self._stream_active = threading.Event() # Signals when stream is actually capturing

        self._transcription_thread: Optional[threading.Thread] = None

        try:
            self._setup_ptt_listener()
        except RuntimeError as e: # Catch specific error from _setup_ptt_listener
            logger.critical(f"Failed to set up PTT listener: {e}", exc_info=True)
            raise # Re-raise to signal critical failure
        
        self._start_stream_thread()
        if not self._stream_active.wait(timeout=5.0): # Wait for stream to become active
            logger.error("Audio stream did not become active during initialization.")

    def _load_model(self) -> WhisperModel:
        try:
            logger.info(f"Loading FasterWhisper model: {STT_MODEL_SIZE} (Device: {STT_DEVICE}, Compute: {STT_COMPUTE_TYPE})")
            model = WhisperModel(STT_MODEL_SIZE, device=STT_DEVICE, compute_type=STT_COMPUTE_TYPE)
            logger.info("FasterWhisper model loaded successfully.")
            return model
        except Exception as e: # Catch specific exceptions like model not found, CUDALibs error etc.
            logger.error(f"Fatal: Failed to load FasterWhisper model '{STT_MODEL_SIZE}'. Error: {e}", exc_info=True)
            raise RuntimeError(f"STT Model loading failed ({STT_MODEL_SIZE}). Check logs and model availability.") from e

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        if status:
            logger.warning(f"Audio callback status: {status}")
        with self._lock:
            if self._ptt_pressed:
                self._audio_buffer.append(indata.copy())

    def _manage_audio_stream(self):
        stream = None
        while not self._stream_stop_event.is_set():
            try:
                if constants.DEBUG_MODE: logger.debug("Attempting to start audio stream...")
                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype=DTYPE,
                    blocksize=BLOCK_SIZE,
                    callback=self._audio_callback
                )
                with stream:
                    self._stream_active.set() 
                    logger.info(f"Audio stream started. Ready for PTT ('{self.ptt_key}').")
                    while stream.active and not self._stream_stop_event.is_set():
                        time.sleep(0.1) # Keep thread alive while stream is active
                    if constants.DEBUG_MODE: logger.debug("Exiting audio stream active loop...")
                
                # If loop exited because stream became inactive but not due to stop_event
                if not self._stream_stop_event.is_set():
                    logger.warning("Audio stream became inactive unexpectedly.")

            except sd.PortAudioError as pae:
                logger.error(f"PortAudio error in stream: {pae}. Check audio devices.")
                self._stream_active.clear() # Ensure it's clear if stream fails
                logger.warning("PortAudioError occurred. Will retry stream setup after a delay.")
            except Exception as e:
                logger.error(f"Unexpected error in audio stream thread: {e}", exc_info=True)
                self._stream_active.clear()
            finally:
                self._stream_active.clear() # Stream is no longer active
                if stream and not stream.closed:
                    try:
                        stream.stop()
                        stream.close()
                        if constants.DEBUG_MODE: logger.debug("Stream stopped and closed in finally block.")
                    except Exception as e_close:
                        logger.warning(f"Error closing stream in finally block: {e_close}", exc_info=constants.DEBUG_MODE)
                stream = None

            if not self._stream_stop_event.is_set():
                 logger.info("Audio stream stopped/failed. Retrying in 5 seconds...")
                 time.sleep(5)
        
        self._stream_active.clear()
        logger.info("Audio stream management thread finished.")

    def _start_stream_thread(self):
        if self._stream_thread is None or not self._stream_thread.is_alive():
            self._stream_stop_event.clear()
            self._stream_active.clear() # Ensure initially clear
            self._stream_thread = threading.Thread(target=self._manage_audio_stream, daemon=True)
            self._stream_thread.start()
            logger.info("Audio stream management thread started.")
        else:
            if constants.DEBUG_MODE: logger.debug("Audio stream management thread already running.")

    def _on_ptt_press(self, event=None): # Pylint: disable=unused-argument
        with self._lock:
            if not self._ptt_pressed:
                logger.info(f"'{self.ptt_key}' pressed, recording started...")
                self._ptt_pressed = True
                self._audio_buffer = [] 
                try: # Clear any stale result from the queue
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass

    def _on_ptt_release(self, event=None):  # Pylint: disable=unused-argument
        audio_to_process = None
        with self._lock:
            if self._ptt_pressed:
                logger.info(f"'{self.ptt_key}' released, preparing for transcription...")
                self._ptt_pressed = False
                if self._audio_buffer:
                    audio_to_process = np.concatenate(self._audio_buffer, axis=0)
                self._audio_buffer = [] # Clear buffer immediately

        if audio_to_process is not None and audio_to_process.size > (SAMPLE_RATE * 0.1): # Min audio length (e.g. 0.1s)
             if self._transcription_thread and self._transcription_thread.is_alive():
                 logger.warning("Previous transcription is still running. Discarding new request.")
                 self._put_result(f"{constants.ERROR_MESSAGE_PREFIX} Transcription overlapped") # Use constant
             else:
                if constants.DEBUG_MODE:
                    logger.debug(f"Starting transcription for {len(audio_to_process)/SAMPLE_RATE:.2f}s audio...")
                self._transcription_thread = threading.Thread(
                    target=self._perform_transcription,
                    args=(audio_to_process,),
                    daemon=True 
                )
                self._transcription_thread.start()
        else:
             logger.info("PTT released, but no (or very little) audio recorded.")
             self._put_result("") # Empty string for no speech

    def _perform_transcription(self, audio_np: np.ndarray):
        if not self._model:
            logger.error("Whisper model not loaded. Cannot transcribe.")
            self._put_result(f"{constants.ERROR_MESSAGE_PREFIX} STT Model not available")
            return

        transcription_text = f"{constants.ERROR_MESSAGE_PREFIX} Transcription failed" # Default error
        try:
            audio_float32 = audio_np.astype(np.float32) / 32768.0 # Normalize
            if audio_float32.ndim > 1 and audio_float32.shape[1] == 1: # Ensure mono
                audio_float32 = audio_float32.squeeze()
            elif audio_float32.ndim > 1:
                logger.warning(f"Audio data has multiple channels ({audio_float32.shape[1]}), taking first channel.")
                audio_float32 = audio_float32[:,0]

            if audio_float32.size == 0:
                logger.warning("Audio data is empty after processing. Skipping transcription.")
                transcription_text = ""
            else:
                if constants.DEBUG_MODE:
                    logger.debug(f"Transcribing audio of size {audio_float32.size} samples ({audio_float32.size / SAMPLE_RATE:.2f}s)")
                
                segments, info = self._model.transcribe(
                    audio_float32, 
                    language=self.language_code, # Pass None for auto-detect
                    beam_size=self.beam_size,
                )
                transcription_text = "".join(segment.text for segment in segments).strip()
                logger.info(f"Transcription: '{transcription_text[:70]}{'...' if len(transcription_text)>70 else ''}' (Lang: {info.language} p={info.language_probability:.2f})")
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            # transcription_text remains the default error message
        finally:
            self._put_result(transcription_text)
            if constants.DEBUG_MODE: logger.debug("Transcription thread finished.")

    def _put_result(self, text: str):
        try:
            # Clear the queue (maxsize 1) before putting new item
            try: self._result_queue.get_nowait()
            except queue.Empty: pass
            self._result_queue.put_nowait(text)
            if constants.DEBUG_MODE: logger.debug(f"Put result in queue: '{text[:50]}...'")
        except queue.Full: # Should be rare with maxsize 1 and prior get_nowait
            logger.error("Result queue unexpectedly full. Race condition or logic error?")

    def _setup_ptt_listener(self):
        try:
            keyboard.on_press_key(self.ptt_key, self._on_ptt_press, suppress=False)
            keyboard.on_release_key(self.ptt_key, self._on_ptt_release, suppress=False)
            logger.info(f"PTT hotkeys ('{self.ptt_key}') registered successfully.")
        except ImportError:
            logger.error("Fatal: The 'keyboard' library is not installed or cannot be accessed.", exc_info=True)
            logger.error("Ensure 'keyboard' is installed. If on Linux, it might need root/sudo or specific uinput permissions.")
            raise RuntimeError("Missing 'keyboard' dependency or insufficient permissions.")
        except Exception as e: # Catch other errors like key not found, OS issues
            logger.error(f"Fatal: Failed to register PTT hotkey '{self.ptt_key}'. Error: {e}", exc_info=True)
            raise RuntimeError(f"Hotkey registration failed for '{self.ptt_key}'. Error: {e}")


    def get_transcription(self, timeout: Optional[float] = 30.0) -> str:
        if not self._model: # Model failed to load
            return f"{constants.ERROR_MESSAGE_PREFIX} STT Model unavailable"
        if not self._stream_active.is_set() and (self._stream_thread is None or not self._stream_thread.is_alive()):
            logger.warning("Audio stream is not active and thread is not running. Attempting to restart stream.")
            self._start_stream_thread() # Attempt to restart
            if not self._stream_active.wait(timeout=5.0): # Wait again
                 logger.error("Audio stream failed to become active. Cannot get transcription.")
                 return f"{constants.ERROR_MESSAGE_PREFIX} Audio stream inactive"
        elif not self._stream_active.is_set():
             logger.warning("Audio stream thread is running but stream is not marked active. Waiting briefly...")
             if not self._stream_active.wait(timeout=2.0):
                 logger.error("Audio stream still not active. Cannot get transcription.")
                 return f"{constants.ERROR_MESSAGE_PREFIX} Audio stream not ready"


        logger.info(f"Ready. Press and hold '{self.ptt_key}', release to transcribe. (Timeout: {timeout}s)")
        try:
            # This blocks until PTT is pressed and released, or timeout
            result = self._result_queue.get(timeout=timeout)
            if constants.DEBUG_MODE: logger.debug(f"Transcription received from queue: '{result[:50]}...'")
            return result
        except queue.Empty:
            logger.warning(f"Timeout ({timeout}s) waiting for transcription result (PTT not used or transcription delayed).")
            return f"{constants.ERROR_MESSAGE_PREFIX} Timeout waiting for transcription"

    def close(self):
        logger.info("Cleaning up STT engine...")
        try:
            keyboard.unhook_all()
            logger.info("Keyboard hooks removed.")
        except Exception as e: # keyboard lib might raise if already unhooked or other issues
            logger.warning(f"Error removing keyboard hooks (may be normal if already unhooked): {e}", exc_info=constants.DEBUG_MODE)

        self._stream_stop_event.set()
        if self._stream_thread and self._stream_thread.is_alive():
            if constants.DEBUG_MODE: logger.debug("Waiting for audio stream thread to join...")
            self._stream_thread.join(timeout=2.0) 
            if self._stream_thread.is_alive():
                logger.warning("Audio stream thread did not stop gracefully.")
        
        if self._transcription_thread and self._transcription_thread.is_alive():
             if constants.DEBUG_MODE: logger.debug("Waiting for potential ongoing transcription thread to join...")
             self._transcription_thread.join(timeout=2.0) 
             if self._transcription_thread.is_alive():
                 logger.warning("Transcription thread still active during cleanup.")
        
        with self._lock: # Clear buffer just in case
            self._audio_buffer = []
        while not self._result_queue.empty(): # Clear queue
            try: self._result_queue.get_nowait()
            except queue.Empty: break
        logger.info("STT engine cleanup complete.")