import os
import json
import time
from colorama import init, Fore, Back, Style
from gpt4all import GPT4All

class AI_Data:
    name : str
    age : int
    type : str
    prompt : str
    temperature : float
    max_tokens : int
    top_p : float
    repeat_penalty : float

def init_variables():
    try:
        with open(os.path.dirname(__file__) + "\config.json", "r") as json_file:
            data = json.load(json_file)
    except:
        print("Unable to open JSON file. File is not found or can be corrupted.")
        exit()
    
    # 0 - EvilChan, 1 - Hinata
    modelNumber = 0
    aidata = AI_Data()
    aidata.name = data["AI_data"][modelNumber]["model_name"]
    aidata.age = data["AI_data"][modelNumber]["model_age"]
    aidata.type = data["AI_data"][modelNumber]["model_type"]
    aidata.prompt = data["AI_data"][modelNumber]["sysprompt"]
    aidata.temperature = data["AI_data"][modelNumber]["temperature"]
    aidata.max_tokens = data["AI_data"][modelNumber]["max_tokens"]
    aidata.top_p = data["AI_data"][modelNumber]["top_p"]
    aidata.repeat_penalty = data["AI_data"][modelNumber]["repeat_penalty"]
    return aidata


class WaifuAI:
    data : AI_Data
    model = GPT4All(os.path.dirname(__file__) + 
                    "\models\llama-2-7b-chat.ggmlv3.q4_0.bin")
    def __init__(self):
        self.data = init_variables()

# TOKEN = os.getenv("DISCORD_TOKEN")
# GUILD = os.getenv("DISCORD_GUILD")
init(autoreset=True)

AwwWaifuAI = WaifuAI()
with AwwWaifuAI.model.chat_session():
    output = AwwWaifuAI.model.generate(prompt="Tell me about yourself", max_tokens=AwwWaifuAI.data.max_tokens)
print(Fore.CYAN + output)