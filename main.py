import ollama
from pygame import mixer
import subprocess
import aidata
import modules.livewhisper as lw
import modules.silero_tts as silero


# Takes text as an input same for output
def textIO(AwwWaifu):
    while True:
        your_prompt = str(input('YOU: '))
        print('AI: ', end='')
        AwwWaifu.stream = ollama.chat(
            model=AwwWaifu.model_name,
            messages=[
                {
                    'role': 'user', 'content': your_prompt
                }
            ],
            stream=True,
        )
        for chunk in AwwWaifu.stream:
            print(chunk['message']['content'], end='', flush=True)      
        print()


# Takes voice as an input same for output
def voiceIO(AwwWaifu):
    myTTS = silero.TTS()
    handler = lw.StreamHandler()
    mixer.init()
    while True:
        try:
            your_prompt = handler.listen()
            your_prompt = your_prompt['text']
        except (KeyboardInterrupt, SystemExit): pass          
        print('YOU: ' + your_prompt)
        print('AI: ', end='')
        AwwWaifu.stream = ollama.chat(
            model=AwwWaifu.model_name,
            messages=[
                {
                    'role': 'user', 'content': your_prompt
                }
            ],
            stream=True,
        )
        AI_answer = ''
        for chunk in AwwWaifu.stream:
            AI_answer += chunk['message']['content']
            print(chunk['message']['content'], end='', flush=True)
        print()
        mixer.music.stop()
        mixer.music.unload()
        myTTS.process_audio(AI_answer)
        mixer.music.load(myTTS.output_file)
        mixer.music.play()
        print()


if __name__ == '__main__':
    # this command boots up ollama
    subprocess.run("ollama list")
    conversationType = 2
    AwwWaifuAI = aidata.WaifuAI()

    # Text
    if conversationType == 1:
        textIO(AwwWaifuAI)
    # Voice
    elif conversationType == 2:
        voiceIO(AwwWaifuAI)
