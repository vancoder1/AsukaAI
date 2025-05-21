# ai_model.py
import os
from typing import Generator, Optional
import logging
import ollama
import httpcore
import httpx
from langchain.globals import set_debug, set_verbose
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import FileChatMessageHistory

from utils import logging_config as lc
from utils import json_handler as jh
from utils import constants
from utils.config_loader import load_settings_from_json

if constants.DEBUG_MODE:
    set_debug(True)
    set_verbose(True)

logger = lc.configure_logger(__name__)

# --- Configuration Loading ---
CONFIG_FILE = constants.CONFIG_FILE_PATH

AI_MODEL_CONFIG_DEFS = {
    'LLAMA_MODEL': ('llama.model', 'llama3.2'),
    'SYSTEM_PROMPT': ('llama.system_prompt', 'You are a helpful AI assistant.'),
    'HISTORY_DIR': ('llama.history_dir', 'data/chats'),
    'SESSION_ID': ('llama.session_id', 'asuka_session'),
    'MAX_TOKEN_LIMIT': ('llama.max_token_limit', 8192),
}

ai_json_handler = jh.JsonHandler(CONFIG_FILE)
_config_values = load_settings_from_json(logger, ai_json_handler, AI_MODEL_CONFIG_DEFS, "AIModel")

LLAMA_MODEL = _config_values['LLAMA_MODEL']
SYSTEM_PROMPT = _config_values['SYSTEM_PROMPT']
HISTORY_DIR = _config_values['HISTORY_DIR']
SESSION_ID = _config_values['SESSION_ID']
MAX_TOKEN_LIMIT = _config_values['MAX_TOKEN_LIMIT']


class AIModel:
    def __init__(self,
                 model_name: str = LLAMA_MODEL,
                 system_prompt: Optional[str] = SYSTEM_PROMPT,
                 session_id: str = SESSION_ID,
                 history_dir: str = HISTORY_DIR,
                 max_token_limit: int = MAX_TOKEN_LIMIT,
                 debug: bool = constants.DEBUG_MODE): # Default to global DEBUG_MODE
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.session_id = session_id
        self.history_dir = history_dir
        self.max_token_limit = max_token_limit
        self.debug = debug # Instance-specific debug, defaults to global

        if self.debug:
            # This log will appear if self.debug is true, regardless of Langchain's global setting
            logger.info(f"AIModel instance created with debug ON. (Global DEBUG_MODE: {constants.DEBUG_MODE}, Langchain set_debug: {str(logging.getLogger('langchain').level <= logging.DEBUG)})")

        self.llm = None
        self.memory = None
        self.chain_with_message_history = None
        self._initialize_components()

    def _initialize_components(self):
        try:
            logger.info(f"Initializing AI Model components for model '{self.model_name}' (Instance debug: {self.debug})...")
            self._check_ollama_connection()
            self.llm = ChatOllama(model=self.model_name)
            logger.info("ChatOllama initialized.")
            self.memory = self._initialize_memory()
            logger.info("Conversation memory initialized.")
            self.chain_with_message_history = self._initialize_chain()
            logger.info("Langchain runnable initialized.")
            logger.info("AI Model components initialized successfully.")
        except (ConnectionError, ollama.ResponseError, RuntimeError) as e:
            logger.error(f"Fatal: Failed to initialize AI Model components: {e}", exc_info=True)
            raise RuntimeError(f"AI Model initialization failed ({e}). Check Ollama service and model availability.") from e
        except Exception as e:
             logger.error(f"Fatal: An unexpected error occurred during AI Model initialization: {e}", exc_info=True)
             raise RuntimeError(f"Unexpected error during AI Model initialization ({e}).") from e

    def _check_ollama_connection(self):
        logger.info(f"Checking connection to Ollama and availability of model '{self.model_name}'...")
        try:
            ollama.generate(model=self.model_name, prompt=".") 
            logger.info(f"Ollama connection successful and model '{self.model_name}' is available.")
        except ollama.ResponseError as re:
            logger.warning(f'Ollama ResponseError: {re.error} (Status: {re.status_code})')
            if re.status_code == 404:
                logger.info(f"Model '{self.model_name}' not found locally. Attempting to pull...")
                self._pull_model()
            else:
                raise RuntimeError(f"Ollama API error: {re.error}") from re
        except (httpcore.ConnectError, httpx.ConnectError, ConnectionRefusedError) as ce:
            logger.error(f'Ollama Connection Error: {ce}')
            raise ConnectionError("Failed to connect to Ollama service. Ensure Ollama is running.") from ce
        except Exception as e:
             logger.error(f"Unexpected error during Ollama check for model '{self.model_name}': {e}", exc_info=True)
             raise RuntimeError("Unexpected error checking Ollama connection.") from e

    def _pull_model(self) -> None:
        logger.info(f"Pulling Ollama model '{self.model_name}'. This may take some time...")
        try:
            current_digest = ""
            for progress in ollama.pull(self.model_name, stream=True):
                digest = progress.get("digest", "")
                if digest != current_digest and digest != "":
                    if self.debug: logger.debug(f"Pulling layer {digest}...")
                    current_digest = digest
                status = progress.get("status", "")
                if status:
                     if "pulling" in status or "downloading" in status:
                          if progress.get("completed") and progress.get("total"):
                                percent = round(progress['completed'] / progress['total'] * 100, 1)
                                print(f"\rStatus: {status} - {percent}%", end="")
                          else:
                                print(f"\rStatus: {status}...", end="")
                     else:
                          print(f"\rStatus: {status}") 
            print() 
            logger.info(f"Model '{self.model_name}' pulled successfully.")
            ollama.generate(model=self.model_name, prompt=".")
            logger.info(f"Model '{self.model_name}' confirmed available after pull.")
        except ollama.ResponseError as ep:
            logger.error(f'Error pulling model "{self.model_name}": {ep.error}')
            raise RuntimeError(f'Failed to pull Ollama model "{self.model_name}": {ep.error}') from ep
        except Exception as e:
            logger.error(f'Unexpected error pulling model "{self.model_name}": {e}', exc_info=True)
            raise RuntimeError(f'Unexpected error pulling Ollama model "{self.model_name}".') from e

    def _initialize_memory(self):
        history_file_path = self._get_history_file_path(self.session_id)
        logger.info(f"Initializing conversation memory with history file: {history_file_path}")
        if self.debug:
            logger.debug(f"Memory summarization trigger token limit: {self.max_token_limit}")
        os.makedirs(os.path.dirname(history_file_path), exist_ok=True)
        file_history = FileChatMessageHistory(history_file_path)
        
        return ConversationSummaryBufferMemory(
            llm=self.llm,
            chat_memory=file_history,
            max_token_limit=self.max_token_limit, 
            return_messages=True,
            memory_key="history", 
            input_key="input"     
        )

    def _get_history_file_path(self, session_id: str) -> str:
        os.makedirs(self.history_dir, exist_ok=True)
        return os.path.join(self.history_dir, f"{session_id}.json")

    def _initialize_chain(self):
        if self.debug: logger.debug("Setting up Langchain prompt and runnable...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"), 
            ("human", "{input}"), 
        ])
        base_chain = prompt | self.llm | StrOutputParser()
        chain_with_history = RunnableWithMessageHistory(
            base_chain,
            lambda session_id_param: self.memory.chat_memory,
            input_messages_key="input", 
            history_messages_key="history", 
        )
        if self.debug: logger.debug("Langchain runnable setup complete.")
        return chain_with_history

    def stream(self, input_text: str) -> Generator[str, None, None]:
        if not self.chain_with_message_history:
             logger.error("Cannot stream response: AI chain not initialized.")
             yield f"{constants.ERROR_MESSAGE_PREFIX} AI Chain not ready"
             return 

        log_prefix = f"AIModel {'(debug ON) ' if self.debug else ''}"
        logger.info(f"{log_prefix}Streaming response for input: '{input_text[:50]}...'")
        try:
            for chunk_idx, chunk_content in enumerate(self.chain_with_message_history.stream(
                {"input": input_text},
                config={"configurable": {"session_id": self.session_id}} # Pass session_id for history
            )):
                # Assuming chunk_content is the actual string chunk
                if self.debug and chunk_idx < 5: 
                    logger.debug(f"Stream chunk {chunk_idx}: '{str(chunk_content)[:50]}...'")
                yield str(chunk_content) # Ensure it's a string
        except Exception as e:
             logger.error(f"Error during AI model streaming: {e}", exc_info=True)
             yield f"{constants.ERROR_MESSAGE_PREFIX} Streaming failed due to an internal error."
        finally:
             if self.debug:
                  logger.debug(f"{log_prefix}Streaming finished.")

    def close(self):
        # Add any explicit cleanup for Ollama, Langchain resources if needed
        logger.info("AIModel close called (placeholder - Langchain/Ollama manage most resources).")
        if self.memory and hasattr(self.memory, 'clear'):
             if self.debug: logger.debug("Clearing AI conversation memory.")
             try:
                self.memory.clear() # Example if memory has a clear method
             except Exception as e:
                logger.warning(f"Error clearing AI memory: {e}", exc_info=self.debug)