import os
import json
import time
from colorama import init, Fore, Back, Style
from gpt4all import GPT4All
import speech_recognition
class AI_Data:
    name : str
    age : int
    type : str
    system_template : str
    temperature : float
    max_tokens : int
    top_p : float
    repeat_penalty : float

def init_variables():
    try:
        with open(os.path.dirname(__file__) + "\config.json", "r") as json_file:
            data = json.load(json_file)
    except:
        print("Unable to open JSON file. File was not found or can be corrupted.")
        exit()
    
    # 0 - EvilChan, 1 - Hinata
    modelNumber = 1
    aidata = AI_Data()
    aidata.name = data["AI_data"][modelNumber]["model_name"]
    aidata.age = data["AI_data"][modelNumber]["model_age"]
    aidata.type = data["AI_data"][modelNumber]["model_type"]
    aidata.system_template = data["AI_data"][modelNumber]["system_template"]
    aidata.temperature = data["AI_data"][modelNumber]["temperature"]
    aidata.max_tokens = data["AI_data"][modelNumber]["max_tokens"]
    aidata.top_p = data["AI_data"][modelNumber]["top_p"]
    aidata.repeat_penalty = data["AI_data"][modelNumber]["repeat_penalty"]
    return aidata

def speechToText():
    pass

class WaifuAI:
    data : AI_Data
    model : GPT4All
    def __init__(self):
        self.data = init_variables()
        self.model = GPT4All(os.path.dirname(__file__) + 
                    "\models\llama-2-7b-chat.ggmlv3.q4_0.bin")


def main():
    # TOKEN = os.getenv("DISCORD_TOKEN")
    # GUILD = os.getenv("DISCORD_GUILD")
    init(autoreset=True)

    AwwWaifuAI = WaifuAI()
    print("Enter \"exit\" to stop the conversation")
    conversation_flag = True
    with AwwWaifuAI.model.chat_session(system_prompt=AwwWaifuAI.data.system_template):
        while conversation_flag:
            tokens = []
            your_prompt = str(input(Fore.CYAN + "YOU: "))
            if (str.lower(your_prompt) == "exit"):
                conversation_flag = False
                break
            for token in AwwWaifuAI.model.generate(prompt=your_prompt, max_tokens=AwwWaifuAI.data.max_tokens, streaming=True):
                tokens.append(token)
                print(Fore.LIGHTRED_EX + str(token), end='')
            print()

if __name__ == '__main__':
    main()