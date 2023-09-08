import os
import json
from gpt4all import GPT4All

class AI_Data:
    #init local variables for AI_Data class from config.json file
    def __init__(self):
        try:
            with open(os.path.dirname(__file__) + "\config.json", "r") as json_file:
                data = json.load(json_file)
        except:
            print("Unable to open JSON file. File was not found or can be corrupted.")
            exit()

        # 0 - EvilChan(Usually rude and toxic), 1 - Hinata(Your true waifu, speaks like you are a couple),
        # 2 - LalaChan(Will speak with you like you are friends)
        modelNumber = 1
        self.name = data["AI_data"][modelNumber]["model_name"]
        self.age = data["AI_data"][modelNumber]["model_age"]
        self.type = data["AI_data"][modelNumber]["model_type"]
        self.system_template = data["AI_data"][modelNumber]["system_template"]
        self.temperature = data["AI_data"][modelNumber]["temperature"]
        self.max_tokens = data["AI_data"][modelNumber]["max_tokens"]
        self.top_p = data["AI_data"][modelNumber]["top_p"]
        self.repeat_penalty = data["AI_data"][modelNumber]["repeat_penalty"]

class WaifuAI:
    def __init__(self):
        self.data = AI_Data()
        self.model = GPT4All(os.path.dirname(__file__) + 
                    "\models\llama-2-7b-chat.ggmlv3.q4_0.bin", allow_download=False)