import ollama
import httpcore
import httpx

class WaifuAI:
    def __init__(self):
        try:
            self.stream = ollama.generate(
                model=self.model_name
            )

        except ollama.ResponseError as re:
            print('Error:', re.error)
            if re.status_code == 404:
                print("Try pulling " + self.model_name + "...")
            try:
                ollama.pull(self.model_name)
            except ollama.ResponseError as ep:
                print("Error:", ep.error)
            else:
                print(self.model_name + " is successfully installed!")
        except (httpcore.ConnectError, httpx.ConnectError) as ce:
            print('Error:', ce)
            print('It seems that ollama is not running on your device, try starting it up first!')
            exit()
