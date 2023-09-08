import os
from colorama import init, Fore, Style
import aidata
import livewhisper

# Takes text as an input same for output
def textIO(AwwWaifu):
    with AwwWaifu.model.chat_session(system_prompt=AwwWaifu.data.system_template):
        while True:
            tokens = []
            your_prompt = str(input(Fore.CYAN + "YOU: "))           
            if (str.lower(your_prompt) == "exit"):
                exit()
            for token in AwwWaifu.model.generate(prompt=your_prompt, max_tokens=AwwWaifu.data.max_tokens, streaming=True):
                tokens.append(token)
                print(Fore.LIGHTRED_EX + str(token), end='')
            print()

# Takes voice as an input same for output
def voiceIO(AwwWaifu):
    with AwwWaifu.model.chat_session(system_prompt=AwwWaifu.data.system_template):
        handler = livewhisper.StreamHandler()
        while True:
            try:
                your_prompt = handler.listen()
                your_prompt = your_prompt['text']
            except (KeyboardInterrupt, SystemExit): pass
            tokens = []           
            print(Fore.CYAN + "YOU: " + your_prompt)
            
            if (str.lower(your_prompt) == "exit"):
                if os.path.exists('dictate.wav'): os.remove('dictate.wav')
                exit()

            for token in AwwWaifu.model.generate(prompt=your_prompt, max_tokens=AwwWaifu.data.max_tokens, streaming=True):
                tokens.append(token)
                print(Fore.LIGHTRED_EX + str(token), end='')
            print()

def main():
    # TOKEN = os.getenv("DISCORD_TOKEN")
    # GUILD = os.getenv("DISCORD_GUILD")
    init(autoreset=True)

    print(Fore.LIGHTGREEN_EX + Style.BRIGHT + "Now choose the type of comunicatiton:")
    print(Fore.LIGHTGREEN_EX + "1. Text conversation")
    print(Fore.LIGHTGREEN_EX + "2. Voice conversation")
    convTypeChoice = int(input(Fore.LIGHTGREEN_EX + "Type a number of your choice: "))
    if (convTypeChoice < 1 or convTypeChoice > 2):
        convTypeChoice = 1

    # AI code
    AwwWaifuAI = aidata.WaifuAI()
    print(Fore.MAGENTA + Style.BRIGHT + "You will speak with " + AwwWaifuAI.data.name)
    print(Fore.MAGENTA + Style.BRIGHT + "Enter or say \"exit\" to stop the conversation\n")

    # Text input
    if (convTypeChoice == 1):
        textIO(AwwWaifuAI)
            
    # Voice input
    elif(convTypeChoice == 2):
        voiceIO(AwwWaifuAI)

if __name__ == '__main__':
    main()