import os
import time
import warnings
from kokoro import KPipeline
import sounddevice as sd
import numpy as np
import utils.logging_config as lc
import utils.json_handler as jh

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = lc.configure_logger(__name__)

# --- Configuration Loading ---
CONFIG_FILE = 'config/config.json'
DEFAULT_SETTINGS = {
    'tts.lang_code': 'en', # Default language code
    'tts.voice': 'af_heart', # Default voice
}

try:
    json_handler = jh.JsonHandler(CONFIG_FILE)
    LANG_CODE = json_handler.get_setting('tts.lang_code', DEFAULT_SETTINGS['tts.lang_code'])
    VOICE = json_handler.get_setting('tts.voice', DEFAULT_SETTINGS['tts.voice'])
except FileNotFoundError:
    logger.warning(f"{CONFIG_FILE} not found. Using default TTS settings.")
    LANG_CODE = DEFAULT_SETTINGS['tts.lang_code']
    VOICE = DEFAULT_SETTINGS['tts.voice']
except Exception as e:
    logger.error(f"Error reading {CONFIG_FILE} for TTS settings: {e}. Using default TTS settings.")
    LANG_CODE = DEFAULT_SETTINGS['tts.lang_code']
    VOICE = DEFAULT_SETTINGS['tts.voice']


# --- Constants ---
# NOTE: These audio parameters are assumed based on common TTS model outputs.
# Verify these values if experiencing audio playback issues (e.g., incorrect speed, pitch, noise).
# Kokoro's documentation or inspecting the output array's properties might be necessary.
SAMPLE_RATE = 24000  # Assumed sample rate in Hz
CHANNELS = 1         # Assumed mono audio
AUDIO_DTYPE = np.float32 # Assumed data type (Kokoro likely outputs float32)

class TTS:
    """Handles Text-to-Speech synthesis and playback using KokoroTTS."""
    def __init__(self,
                 lang_code: str = LANG_CODE,
                 voice: str = VOICE):
        """
        Initializes the TTS engine.

        Args:
            lang_code (str): Language code for TTS (e.g., 'en', 'es').
            voice (str): Specific voice model/speaker ID to use.
        """
        self.lang_code = lang_code
        self.voice = voice
        self.pipeline = self._initialize_pipeline()
        logger.info(f"TTS initialized with lang_code='{self.lang_code}', voice='{self.voice}'")
        logger.info(f"Audio playback configured for {SAMPLE_RATE} Hz, {CHANNELS} channel(s), dtype='{AUDIO_DTYPE}'")


    def _initialize_pipeline(self) -> KPipeline:
        """Loads the KokoroTTS pipeline."""
        try:
            # Corrected repo_id based on common Kokoro usage, adjust if needed
            repo_id = 'hexgrad/Kokoro-82M'
            logger.info(f"Initializing KokoroTTS pipeline (repo='{repo_id}', lang='{self.lang_code}')...")
            start_time = time.time()
            # Pass lang_code during initialization if supported, otherwise it might be inferred or set later
            pipeline = KPipeline(repo_id=repo_id, lang_code=self.lang_code) # Adjust if lang_code needed here
            end_time = time.time()
            logger.info(f"KokoroTTS pipeline initialized successfully in {end_time - start_time:.2f} seconds.")
            return pipeline
        except Exception as e:
            logger.error(f'Fatal: Failed to initialize KokoroTTS pipeline: {e}', exc_info=True)
            raise RuntimeError("TTS Pipeline initialization failed.") from e

    def stream_inference(self, text: str, speed: float = 1.0): # Default speed to 1.0
        """
        Generates audio from text and plays it back in real-time using sounddevice.

        Args:
            text (str): The text to synthesize.
            speed (float): The speed factor for speech generation (1.0 is normal).
        """
        if not text.strip():
            logger.warning("stream_inference called with empty text.")
            return

        logger.info(f"Starting streaming inference (Speed: {speed}) for text: '{text[:50]}...'")

        try:
            # Create the generator
            # Ensure the pipeline call matches Kokoro's expected arguments
            generator = self.pipeline(
                text=text,
                voice=self.voice,
                speed=speed,
                split_pattern=r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
            )

            # Set up the audio stream using a context manager for proper cleanup
            with sd.OutputStream(samplerate=SAMPLE_RATE,
                                 channels=CHANNELS,
                                 dtype=AUDIO_DTYPE) as stream:
                logger.debug(f"Audio stream opened ({stream.samplerate} Hz, {stream.channels} channels, {stream.dtype})")

                start_time = time.time()
                total_samples = 0
                first_chunk_received = False

                # Iterate through the generated audio chunks
                # Adjust iteration based on what the Kokoro generator actually yields
                # It might yield only audio chunks, or tuples like (text_segment, audio_chunk)
                for i, (gs, ps, audio_chunk) in enumerate(generator): # Assuming it yields tuples: (segment_info, audio_chunk)
                    # Example if it yields tuples: for i, (segment_info, audio_chunk) in enumerate(generator):
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        if not first_chunk_received:
                             logger.debug("First audio chunk received.")
                             first_chunk_received = True

                        # Ensure audio is a NumPy array
                        if not isinstance(audio_chunk, np.ndarray):
                           audio_chunk = np.array(audio_chunk, dtype=AUDIO_DTYPE)

                        # Ensure correct dtype
                        if audio_chunk.dtype != np.dtype(AUDIO_DTYPE):
                            logger.warning(f"Audio chunk {i} has dtype {audio_chunk.dtype}, converting to {AUDIO_DTYPE}")
                            audio_chunk = audio_chunk.astype(AUDIO_DTYPE)

                        # Play the audio chunk
                        stream.write(audio_chunk)
                        total_samples += len(audio_chunk)
                        logger.debug(f"Played chunk {i}, length: {len(audio_chunk)} samples ({len(audio_chunk)/SAMPLE_RATE:.2f}s)")
                    else:
                        logger.debug(f"Received empty or None audio chunk {i}")

                end_time = time.time()

                if total_samples > 0:
                    total_duration = total_samples / SAMPLE_RATE
                    processing_time = end_time - start_time
                    logger.info(f"Streaming finished. Total audio duration: {total_duration:.2f}s. Processing time: {processing_time:.2f}s.")

                    # Wait for the stream buffer to finish playing before exiting.
                    # This helps prevent the sound from being cut off abruptly.
                    # sd.sleep takes duration in milliseconds.
                    logger.debug(f"Waiting for audio buffer to clear (latency: {stream.latency:.3f}s)...")
                    sd.sleep(int(stream.latency * 1000 * 1.5)) # Wait 1.5x latency for safety
                    logger.debug("Buffer clear wait finished.")
                else:
                     logger.info("Streaming finished. No audio data was generated or played.")


        except sd.PortAudioError as pae:
             logger.error(f"PortAudioError during streaming: {pae}", exc_info=True)
             # Handle specific audio device errors if possible
        except Exception as e:
            logger.error(f"Error during streaming inference for text '{text[:50]}...': {e}", exc_info=True)
        finally:
            logger.debug("Stream inference function finished.")

    def close(self):
        """Placeholder for cleaning up TTS resources if needed in the future."""
        logger.info("Closing TTS engine resources (placeholder)...")
        # Add cleanup logic here if the pipeline or other resources need explicit closing
        # For example: del self.pipeline
        pass
