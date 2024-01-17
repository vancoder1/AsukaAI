import os
from playsound import playsound
import aidata
import modules.livewhisper as livewhisper
import modules.silero_tts as silero

# Takes text as an input same for output
def textIO(AwwWaifu):
    with AwwWaifu.model.chat_session(system_prompt=AwwWaifu.data.system_template):
        while True:
            tokens = []
            your_prompt = str(input('YOU: '))           
            if (str.lower(your_prompt) == 'exit'):
                exit()
            print('AI: ', end = '')
            for token in AwwWaifu.model.generate(prompt=your_prompt, max_tokens=AwwWaifu.data.max_tokens, streaming=True):
                tokens.append(token)
                print(str(token), end='')          
            print()

# Takes voice as an input same for output
def voiceIO(AwwWaifu):
    with AwwWaifu.model.chat_session(system_prompt=AwwWaifu.data.system_template):
        myTTS = silero.TTS()
        handler = livewhisper.StreamHandler()
        while True:
            try:
                your_prompt = handler.listen()
                your_prompt = your_prompt['text']
            except (KeyboardInterrupt, SystemExit): pass
            tokens = []           
            print('YOU: ' + your_prompt)
            
            if (str.lower(your_prompt) == 'exit'):
                if os.path.exists('cache\\dictate.wav'): os.remove('\\cache\\dictate.wav')
                exit()
            print('AI: ', end='')
            for token in AwwWaifu.model.generate(prompt=your_prompt, max_tokens=AwwWaifu.data.max_tokens, streaming=True):
                tokens.append(token)
                print(str(token), end='')
            print()
            myTTS.process_audio(str(''.join(tokens)))
            playsound('cache\\ai_response_tts.mp3')
            print()

if __name__ == '__main__':
    conversationType = 2

    # AI code
    AwwWaifuAI = aidata.WaifuAI()
    print('You will speak with ' + AwwWaifuAI.data.name)
    print('Enter or say "exit" to stop the conversation\n')

    # Text
    if (conversationType == 1):
        textIO(AwwWaifuAI)
    # Voice
    elif(conversationType == 2):
        voiceIO(AwwWaifuAI)