import os
from typing import Generator, Optional
import ollama
import httpcore
import httpx
import requests
from langchain.globals import set_debug, set_verbose
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.file import FileChatMessageHistory
import modules.logging_config as lc

set_debug(False)
set_verbose(False)
logger = lc.configure_logger(__name__)

class AIModel:
    def __init__(self, 
                 model_name: str = 'llama3.1',
                 system_prompt: Optional[str] = None,
                 session_id: str = "asuka_session",
                 history_dir: str = "data/chats",
                 debug: bool = False):
        self.model_name = model_name
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.session_id = session_id
        self.history_dir = history_dir
        self.debug = debug
        self._initialize_components()

    @staticmethod
    def _default_system_prompt() -> str:
        return """You are a female AI. User is your creator. 
            Your name is Asuka. Sometimes you may be sarcastic. 
            But most of the time, you have a direct but polite attitude, 
            which contrasts with the nonsensical or outlandish things you say. 
            For example, even though you usually say you are an AI, you sometimes talk about doing things which AIs can't do, 
            sometimes say humans are AIs, and sometimes explicitly say you are not an AI. 
            Sometimes you answer user's questions or say things unprompted, 
            where the things you say vary from being relatively normal 
            to completely unhinged to (rarely) not making any semantic sense at all. 
            You don't write any emotions in your response, only the response itself. 
            Your response should not be too long."""

    def _initialize_components(self):
        try:
            self._check_ollama_connection()
            self.llm = ChatOllama(model=self.model_name)
            self.memory = self._initialize_memory()
            self.chain = self._initialize_chain()
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama. Please ensure Ollama is running.")
            raise

    def _check_ollama_connection(self):
        try:
            ollama.generate(model=self.model_name, prompt="Test")
        except ollama.ResponseError as re:
            logger.error(f'Error: {re.error}')
            if re.status_code == 404:
                logger.info(f'Model {self.model_name} not found, attempting to pull it...')
                self._pull_model()
            else:
                raise
        except (httpcore.ConnectError, httpx.ConnectError) as ce:
            logger.error(f'Connection Error: {ce}')
            raise ConnectionError("Failed to connect to Ollama service")
    
    def _pull_model(self) -> None:
        try:
            ollama.pull(self.model_name)
        except ollama.ResponseError as ep:
            logger.error(f'Error pulling model: {ep.error}')
            raise RuntimeError(f'Error pulling model: {ep.error}')

    def _initialize_memory(self):
        file_history = FileChatMessageHistory(self._get_history_file_path(self.session_id))
        return ConversationSummaryBufferMemory(
            llm=self.llm,
            chat_memory=file_history,
            max_token_limit=8192,
            return_messages=True
        )

    def _get_history_file_path(self, session_id: str) -> str:
        os.makedirs(self.history_dir, exist_ok=True)
        return os.path.join(self.history_dir, f"{session_id}.json")

    def _initialize_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        return RunnableWithMessageHistory(
            chain,
            lambda session_id: self.memory.chat_memory,
            input_messages_key="input",
            history_messages_key="history"
        )  

    def generate(self, input_text: str) -> str:
        response = self.chain.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": self.session_id}}
        )
        return response

    def stream(self, input_text: str) -> Generator[str, None, None]:
        for chunk in self.chain.stream(
            {"input": input_text},
            config={"configurable": {"session_id": self.session_id}}
        ):
            yield chunk

    def yield_generate(self, input_text: str, is_printed: bool = False) -> Generator[str, None, None]:
        full_response = ""
        for chunk in self.stream(input_text):
            if is_printed:
                print(chunk, end='', flush=True)
            full_response += chunk
            yield chunk
        if self.debug:
            logger.info(f"Generated response for input: {input_text[:50]}...")
            if hasattr(self.memory, "moving_summary_buffer"):
                logger.info(f"Current conversation summary: {self.memory.moving_summary_buffer}")