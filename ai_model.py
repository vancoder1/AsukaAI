import ollama
import httpcore
import httpx
import modules.logging_config as lf

logger = lf.configure_logger(__name__)

class AIModel:
    def __init__(self, model_name: str = 'Asuka'):
        self.model_name = model_name
        self.stream = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            self.stream = ollama.generate(model=self.model_name)
        except ollama.ResponseError as re:
            logger.error(f'Error: {re.error}')
            if re.status_code == 404:
                logger.info(f'Model {self.model_name} not found, attempting to pull it...')
                self._pull_model()
        except (httpcore.ConnectError, httpx.ConnectError) as ce:
            logger.error(f'Connection Error: {ce}')
            logger.error('It seems that ollama is not running on your device. Please start it up first!')
            exit(1)

    def _pull_model(self):
        try:
            ollama.pull(self.model_name)
        except ollama.ResponseError as ep:
            logger.error(f'Error pulling model: {ep.error}')
            exit(1)
        else:
            logger.info(f'Model {self.model_name} is successfully installed!')