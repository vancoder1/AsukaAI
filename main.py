import os
import json
from colorama import init, Fore, Style
import whisper
from gpt4all import GPT4All
import aidata
       
def speechToText():
    pass

def main():
    # TOKEN = os.getenv("DISCORD_TOKEN")
    # GUILD = os.getenv("DISCORD_GUILD")
    init(autoreset=True)

    # Model and conversation type choosing
    print(Fore.MAGENTA + Style.BRIGHT + "Choose your model:")
    print(Fore.MAGENTA + "1. EvilChan - usually rude and toxic")
    print(Fore.MAGENTA + "2. HinataChan - your true waifu, thinks that you are a couple")
    print(Fore.MAGENTA + "3. LalaChan - will speak with you like you are friends")
    modelChoice = int(input(Fore.MAGENTA + "Type a number of your choice: "))
    print()
    if (modelChoice < 1 or modelChoice > 3):
        modelChoice = 3
    modelChoice -= 1

    print(Fore.LIGHTGREEN_EX + Style.BRIGHT + "Now choose the type of comunicatiton:")
    print(Fore.LIGHTGREEN_EX + "1. Text conversation")
    print(Fore.LIGHTGREEN_EX + "2. Voice conversation")
    convTypeChoice = int(input(Fore.LIGHTGREEN_EX + "Type a number of your choice: "))
    if (convTypeChoice < 1 or convTypeChoice > 2):
        convTypeChoice = 1

    # AI code
    AwwWaifuAI = aidata.WaifuAI(modelChoice)
    print(Fore.MAGENTA + Style.BRIGHT + "You will speak with " + AwwWaifuAI.data.name)
    print(Fore.MAGENTA + Style.BRIGHT + "Enter \"exit\" to stop the conversation\n")
    conversation_flag = True

    with AwwWaifuAI.model.chat_session(system_prompt=AwwWaifuAI.data.system_template):
        while conversation_flag:
            tokens = []
            if (convTypeChoice == 1):
                your_prompt = str(input(Fore.CYAN + "YOU: "))
            elif(convTypeChoice == 2):
                # your_prompt = speechToText()
                print(Fore.CYAN + "YOU: " + your_prompt)
            
            if (str.lower(your_prompt) == "exit"):
                conversation_flag = False
                break

            for token in AwwWaifuAI.model.generate(prompt=your_prompt, max_tokens=AwwWaifuAI.data.max_tokens, streaming=True):
                tokens.append(token)
                print(Fore.LIGHTRED_EX + str(token), end='')
            print()

if __name__ == '__main__':
    main()