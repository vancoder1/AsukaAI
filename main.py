from RealtimeSTT import AudioToTextRecorder
import warnings
import ai_model
import modules.tts_engine as tts_engine
import modules.logging_config as lf
import modules.json_handler as jh

warnings.filterwarnings("ignore", category=UserWarning)
logger = lf.configure_logger(__name__)
json_handler = jh.JsonHandler('config.json')

STT_MODEL = json_handler.get_setting('stt.model')


if __name__ == '__main__':
    print('Initializing Ollama...')
    AsukaAI = ai_model.AIModel()
    print('Initializing TTS...')
    tts = tts_engine.TTS()
    print('Initializing recorder...')
    recorder = AudioToTextRecorder(model=STT_MODEL,
                                   language='en',
                                   compute_type='int8',
                                   silero_use_onnx=True,
                                   spinner=False,)
      
    while True:
        try:
            print('You may speak now...')
            recorder.start()
            print(f'\nYOU: {(user_text := recorder.stop().text())}\n', end="", flush=True)
            print('AI: ', end='')
            tts.stream_inference(AsukaAI.yield_generate(user_text, True))
            print(end='\n')
        except Exception as e:
            logger.error(f"Error during AI response: {e}")
            continue
