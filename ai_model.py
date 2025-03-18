import os
from typing import Generator, Optional
import warnings
import ollama
import httpcore
import httpx
import requests
from langchain.globals import set_debug, set_verbose
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.file import FileChatMessageHistory
import modules.logging_config as lc
import modules.json_handler as jh

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

set_debug(False)
set_verbose(False)
logger = lc.configure_logger(__name__)
json_handler = jh.JsonHandler('config.json')

LLAMA_MODEL = json_handler.get_setting('llama.model')
SYSTEM_PROMPT = json_handler.get_setting('llama.system_prompt')

class AIModel:
    def __init__(self, 
                 model_name: str = LLAMA_MODEL,
                 system_prompt: Optional[str] = SYSTEM_PROMPT,
                 session_id: str = "asuka_session",
                 history_dir: str = "data/chats",
                 debug: bool = False):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.session_id = session_id
        self.history_dir = history_dir
        self.debug = debug
        self._initialize_components()

    def _initialize_components(self):
        try:
            self._check_ollama_connection()
            self.llm = ChatOllama(model=self.model_name)
            self.memory = self._initialize_memory()
            self.chain = self._initialize_chain()
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Ollama: {e}")
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
        chain_with_message_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: self.memory.chat_memory,
            input_messages_key="input",
            history_messages_key="history"
        )
        return (RunnablePassthrough.assign(messages_summarized=self.summarize_messages)
            | chain_with_message_history
        )
    
    def summarize_messages(self, chain_input):
        stored_messages = self.memory.chat_memory.messages
        if len(stored_messages) == 0:
            return False
        summarization_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="history"),
                (
                    "user",
                    "Distill the above chat messages into a single summary message. Include as many specific details as you can.",
                ),
            ]
        )
        summarization_chain = summarization_prompt | self.llm
        summary_message = summarization_chain.invoke({"history": stored_messages})
        self.memory.chat_memory.clear()
        self.memory.chat_memory.add_message(summary_message)
        return True

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