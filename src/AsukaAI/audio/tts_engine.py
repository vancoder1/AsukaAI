import time
import sounddevice as sd
import numpy as np
from typing import Optional
from utils import logging_config as lc
from utils import json_handler as jh
from utils import constants
from utils.config_loader import load_settings_from_json

try:
    from kokoro import KPipeline # Assuming this is correctly installed
except ImportError as e:
    # Log this critical error if logger is already configured, or print and raise
    # This allows the program to at least start and log the problem if KPipeline is missing.
    initial_logger = lc.configure_logger(__name__) # Configure logger early for this
    initial_logger.critical(f"Failed to import KPipeline from kokoro: {e}. TTS will not function.", exc_info=True)
    # KPipeline = None # Or raise an error immediately
    raise RuntimeError(f"KPipeline import failed. TTS unavailable. Error: {e}") from e


logger = lc.configure_logger(__name__) # Logger level will respect DEBUG_MODE

# --- Configuration Loading ---
CONFIG_FILE = constants.CONFIG_FILE_PATH

TTS_CONFIG_DEFS = {
    'LANG_CODE': ('tts.lang_code', 'a'),
    'VOICE': ('tts.voice', 'af_heart'),
}

tts_json_handler = jh.JsonHandler(CONFIG_FILE)
_config_values = load_settings_from_json(logger, tts_json_handler, TTS_CONFIG_DEFS, "TTS")

LANG_CODE = _config_values['LANG_CODE']
VOICE = _config_values['VOICE']

# --- Constants ---
SAMPLE_RATE = 24000
CHANNELS = 1
AUDIO_DTYPE = np.float32

class TTS:
    def __init__(self,
                 lang_code: str = LANG_CODE,
                 voice: str = VOICE):
        self.lang_code = lang_code
        self.voice = voice
        self.pipeline: Optional[KPipeline] = None # Initialize as Optional
        if KPipeline is not None: # Check if import was successful
            self.pipeline = self._initialize_pipeline()
            logger.info(f"TTS initialized with lang_code='{self.lang_code}', voice='{self.voice}'")
            logger.info(f"Audio playback configured for {SAMPLE_RATE} Hz, {CHANNELS} channel(s), dtype='{AUDIO_DTYPE.__name__}'")
        else:
            logger.error("Kokoro KPipeline not available. TTS functionality will be disabled.")


    def _initialize_pipeline(self) -> Optional[KPipeline]:
        if KPipeline is None:
            return None
        try:
            repo_id = 'hexgrad/Kokoro-82M'
            logger.info(f"Initializing KokoroTTS pipeline (repo='{repo_id}', lang='{self.lang_code}')...")
            start_time = time.time()
            pipeline = KPipeline(repo_id=repo_id, lang_code=self.lang_code)
            end_time = time.time()
            logger.info(f"KokoroTTS pipeline initialized successfully in {end_time - start_time:.2f} seconds.")
            return pipeline
        except Exception as e:
            logger.error(f'Fatal: Failed to initialize KokoroTTS pipeline: {e}', exc_info=True)
            # Depending on desired behavior, could raise RuntimeError or allow graceful degradation
            # For now, let's allow degradation, self.pipeline will remain None or error state
            raise RuntimeError("TTS Pipeline initialization failed.") from e # Or return None and handle

    def stream_inference(self, text: str, speed: float = 1.0):
        if not self.pipeline:
            logger.error("TTS pipeline not initialized. Cannot perform inference.")
            # Optionally, provide a silent failure or a specific error sound/message
            return
        if not text.strip():
            logger.warning("stream_inference called with empty text.")
            return

        logger.info(f"Starting streaming inference (Speed: {speed}) for text: '{text[:50]}...'")
        try:
            generator = self.pipeline(
                text=text,
                voice=self.voice,
                speed=speed,
                split_pattern=r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
            )
            with sd.OutputStream(samplerate=SAMPLE_RATE,
                                 channels=CHANNELS,
                                 dtype=AUDIO_DTYPE) as stream:
                if constants.DEBUG_MODE:
                    logger.debug(f"Audio stream opened ({stream.samplerate} Hz, {stream.channels} channels, {stream.dtype})")
                
                start_time = time.time()
                total_samples = 0
                first_chunk_received = False

                for i, (gs, ps, audio_chunk_data) in enumerate(generator):
                    audio_chunk = audio_chunk_data # Assuming audio_chunk_data is the numpy array
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        if not first_chunk_received:
                             if constants.DEBUG_MODE: logger.debug("First audio chunk received.")
                             first_chunk_received = True
                        if not isinstance(audio_chunk, np.ndarray):
                           audio_chunk = np.array(audio_chunk, dtype=AUDIO_DTYPE)
                        if audio_chunk.dtype != AUDIO_DTYPE: # Use direct comparison
                            logger.warning(f"Audio chunk {i} has dtype {audio_chunk.dtype}, converting to {AUDIO_DTYPE.__name__}")
                            audio_chunk = audio_chunk.astype(AUDIO_DTYPE)
                        stream.write(audio_chunk)
                        total_samples += len(audio_chunk)
                        if constants.DEBUG_MODE:
                            logger.debug(f"Played chunk {i}, length: {len(audio_chunk)} samples ({len(audio_chunk)/SAMPLE_RATE:.2f}s)")
                    elif constants.DEBUG_MODE:
                        logger.debug(f"Received empty or None audio chunk {i}")
                
                end_time = time.time()
                if total_samples > 0:
                    total_duration = total_samples / SAMPLE_RATE
                    processing_time = end_time - start_time
                    logger.info(f"Streaming finished. Total audio duration: {total_duration:.2f}s. Processing time: {processing_time:.2f}s.")
                    if constants.DEBUG_MODE: logger.debug(f"Waiting for audio buffer to clear (latency: {stream.latency:.3f}s)...")
                    sd.sleep(int(stream.latency * 1000 * 1.5)) # Wait for buffer to play out
                    if constants.DEBUG_MODE: logger.debug("Buffer clear wait finished.")
                else:
                     logger.info("Streaming finished. No audio data was generated or played.")
        except sd.PortAudioError as pae:
             logger.error(f"PortAudioError during streaming: {pae}", exc_info=True)
             # Potentially raise to inform main loop or try to recover audio device
        except AttributeError as ae: # If self.pipeline is None and somehow used
            logger.error(f"TTS pipeline attribute error (likely not initialized): {ae}", exc_info=constants.DEBUG_MODE)
        except Exception as e:
            logger.error(f"Error during streaming inference for text '{text[:50]}...': {e}", exc_info=True)
        finally:
            if constants.DEBUG_MODE: logger.debug("Stream inference function finished.")

    def close(self):
        logger.info("Closing TTS engine resources...")
        if hasattr(sd, 'stop'): # Stop any sounddevice activity if applicable
            try:
                sd.stop()
                if constants.DEBUG_MODE: logger.debug("Sounddevice stream stopped if active.")
            except Exception as e:
                logger.warning(f"Exception while trying to stop sounddevice: {e}", exc_info=constants.DEBUG_MODE)
        logger.info("TTS engine cleanup complete.")