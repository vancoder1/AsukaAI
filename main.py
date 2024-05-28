import ollama
import aidata
from typing import Any
from pygame import mixer
import modules.coqui_tts as coqui_tts
import modules.fasterwhisper_stt as fw
import modules.logging_config as lf

logger = lf.configure_logger(__name__)

def voiceIO(AsukaAI):
    mixer.init()
    print('Initializing xtts...')
    tts = coqui_tts.Coqui_TTS()
    print('Initializing fasterwhisper...')
    whisper = fw.FasterWhisper()
    print('Initializing push-to-talk recorder...')
    recorder = fw.PushToTalkRecorder()
    recorder.start()

    while True:
        print(f'Press {recorder.key} to start recording and release to stop.')
        recorder.wait_for_recording()
        try:
            your_prompt = whisper.transcribe()
        except (SystemExit):
            logger.info('Exiting...')
            break
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            continue
        
        print(f'YOU: {your_prompt}')

        try:
            AsukaAI.stream = ollama.chat(
                model=AsukaAI.model_name,
                messages=[
                    {
                        'role': 'user', 'content': your_prompt
                    }
                ],
                stream=True,
            )
            print('AI: ', end='')
            AI_answer = ''
            for chunk in AsukaAI.stream:
                AI_answer += chunk['message']['content']
                print(chunk['message']['content'], end='', flush=True)
            print(end='\n')
        except Exception as e:
            logger.error(f"Error during AI response: {e}")
            continue
        
        mixer.music.stop()
        mixer.music.unload()
        
        try:
            tts.process_audio(AI_answer)
            mixer.music.load(tts.output_file)
            mixer.music.play()
        except Exception as e:
            logger.error(f"Error during TTS processing: {e}")


if __name__ == '__main__':
    try:
        # This command boots up ollama
        # subprocess.run("ollama list")
        AsukaAI = aidata.AI()
        voiceIO(AsukaAI)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
