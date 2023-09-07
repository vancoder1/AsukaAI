import os
import json
from gpt4all import GPT4All

class AI_Data:
    name : str
    age : int
    type : str
    system_template : str
    temperature : float
    max_tokens : int
    top_p : float
    repeat_penalty : float

#init local variables for AI_Data class from config.json file
def init_variables(modelNum):
    try:
        with open(os.path.dirname(__file__) + "\config.json", "r") as json_file:
            data = json.load(json_file)
    except:
        print("Unable to open JSON file. File was not found or can be corrupted.")
        exit()
    
    # 0 - EvilChan(Usually rude and toxic), 1 - Hinata(Your true waifu, speaks like you are a couple),
    # 2 - LalaChan(Will speak with you like you are friends)
    modelNumber = modelNum

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

class WaifuAI:
    data : AI_Data
    model : GPT4All
    model_choice : int
    def __init__(self, model_ch):
        self.model_choice = model_ch
        self.data = init_variables(self.model_choice)
        self.model = GPT4All(os.path.dirname(__file__) + 
                    "\models\llama-2-7b-chat.ggmlv3.q4_0.bin", allow_download=False)