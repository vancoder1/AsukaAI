import os
from typing import Generator, Optional, Any
import warnings
import ollama
import httpcore # Keep for potential ollama dependency
import httpx   # Keep for potential ollama dependency
# import requests # Likely unused directly, ollama client handles requests
from langchain.globals import set_debug, set_verbose
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
# Remove RunnablePassthrough as custom summarization is removed
# from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import FileChatMessageHistory

# Use relative imports
from utils import logging_config as lc
from utils import json_handler as jh

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configure Langchain verbosity (can be overridden by debug flag)
set_debug(False)
set_verbose(False)

logger = lc.configure_logger(__name__)

# --- Configuration Loading ---
CONFIG_FILE = 'config/config.json'
DEFAULT_SETTINGS = {
    'llama.model': 'llama3.2', # Example default model
    'llama.system_prompt': 'You are a helpful AI assistant.',
    'llama.history_dir': 'data/chats',
    'llama.session_id': 'asuka_session',
    'llama.max_token_limit': 8192, # Default token limit for memory summarization
}

try:
    json_handler = jh.JsonHandler(CONFIG_FILE)
    LLAMA_MODEL = json_handler.get_setting('llama.model', DEFAULT_SETTINGS['llama.model'])
    SYSTEM_PROMPT = json_handler.get_setting('llama.system_prompt', DEFAULT_SETTINGS['llama.system_prompt'])
    HISTORY_DIR = json_handler.get_setting('llama.history_dir', DEFAULT_SETTINGS['llama.history_dir'])
    SESSION_ID = json_handler.get_setting('llama.session_id', DEFAULT_SETTINGS['llama.session_id'])
    MAX_TOKEN_LIMIT = json_handler.get_setting('llama.max_token_limit', DEFAULT_SETTINGS['llama.max_token_limit'])
    logger.info(f"Loaded AI Model configuration from {CONFIG_FILE} (or defaults).")
except FileNotFoundError:
    logger.warning(f"{CONFIG_FILE} not found. Using default AI Model settings.")
    LLAMA_MODEL = DEFAULT_SETTINGS['llama.model']
    SYSTEM_PROMPT = DEFAULT_SETTINGS['llama.system_prompt']
    HISTORY_DIR = DEFAULT_SETTINGS['llama.history_dir']
    SESSION_ID = DEFAULT_SETTINGS['llama.session_id']
    MAX_TOKEN_LIMIT = DEFAULT_SETTINGS['llama.max_token_limit']
except Exception as e:
    logger.error(f"Error reading {CONFIG_FILE} for AI Model settings: {e}. Using default AI Model settings.", exc_info=True)
    LLAMA_MODEL = DEFAULT_SETTINGS['llama.model']
    SYSTEM_PROMPT = DEFAULT_SETTINGS['llama.system_prompt']
    HISTORY_DIR = DEFAULT_SETTINGS['llama.history_dir']
    SESSION_ID = DEFAULT_SETTINGS['llama.session_id']
    MAX_TOKEN_LIMIT = DEFAULT_SETTINGS['llama.max_token_limit']


class AIModel:
    """
    Manages the connection to an Ollama language model, conversation history,
    and generation/streaming of responses.
    """
    def __init__(self,
                 model_name: str = LLAMA_MODEL,
                 system_prompt: Optional[str] = SYSTEM_PROMPT,
                 session_id: str = SESSION_ID,
                 history_dir: str = HISTORY_DIR,
                 max_token_limit: int = MAX_TOKEN_LIMIT,
                 debug: bool = False):
        """
        Initializes the AI Model components.

        Args:
            model_name (str): The name of the Ollama model to use.
            system_prompt (Optional[str]): The system prompt for the AI.
            session_id (str): Identifier for the conversation session history.
            history_dir (str): Directory to store chat history files.
            max_token_limit (int): Max token limit before ConversationSummaryBufferMemory summarizes.
            debug (bool): Enable debug logging for Langchain components.
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.session_id = session_id
        self.history_dir = history_dir
        self.max_token_limit = max_token_limit
        self.debug = debug

        if self.debug:
            set_debug(True)
            set_verbose(True)
            logger.setLevel(logging.DEBUG)
            logger.info("Debug mode enabled for AIModel and Langchain.")

        self.llm = None
        self.memory = None
        self.chain_with_message_history = None
        self._initialize_components()

    def _initialize_components(self):
        """Initializes Ollama connection, memory, and the Langchain Runnable."""
        try:
            logger.info(f"Initializing AI Model components for model '{self.model_name}'...")
            self._check_ollama_connection() # Check connection and pull model if needed first
            self.llm = ChatOllama(model=self.model_name)
            logger.info("ChatOllama initialized.")
            self.memory = self._initialize_memory()
            logger.info("Conversation memory initialized.")
            self.chain_with_message_history = self._initialize_chain()
            logger.info("Langchain runnable initialized.")
            logger.info("AI Model components initialized successfully.")
        except (ConnectionError, ollama.ResponseError, RuntimeError) as e:
            logger.error(f"Fatal: Failed to initialize AI Model components: {e}", exc_info=True)
            # Re-raise critical errors to prevent application from starting incorrectly
            raise RuntimeError("AI Model initialization failed. Check Ollama service and model availability.") from e
        except Exception as e:
             logger.error(f"Fatal: An unexpected error occurred during AI Model initialization: {e}", exc_info=True)
             raise RuntimeError("Unexpected error during AI Model initialization.") from e


    def _check_ollama_connection(self):
        """Checks connection to Ollama and pulls the model if it's not available."""
        logger.info(f"Checking connection to Ollama and availability of model '{self.model_name}'...")
        try:
            # Use ollama.list() or a lightweight command first if possible
            # ollama.list() # Example check
            # If list doesn't error, try a small generation
            ollama.generate(model=self.model_name, prompt=".") # Minimal prompt
            logger.info(f"Ollama connection successful and model '{self.model_name}' is available.")
        except ollama.ResponseError as re:
            logger.warning(f'Ollama ResponseError: {re.error} (Status: {re.status_code})')
            if re.status_code == 404:
                logger.info(f"Model '{self.model_name}' not found locally. Attempting to pull...")
                self._pull_model()
            else:
                # Re-raise other response errors (e.g., authentication, server issues)
                raise RuntimeError(f"Ollama API error: {re.error}") from re
        except (httpcore.ConnectError, httpx.ConnectError, ConnectionRefusedError) as ce:
            logger.error(f'Ollama Connection Error: {ce}')
            raise ConnectionError("Failed to connect to Ollama service. Ensure Ollama is running.") from ce
        except Exception as e:
             logger.error(f"Unexpected error during Ollama check for model '{self.model_name}': {e}", exc_info=True)
             raise RuntimeError("Unexpected error checking Ollama connection.") from e

    def _pull_model(self) -> None:
        """Pulls the specified Ollama model."""
        logger.info(f"Pulling Ollama model '{self.model_name}'. This may take some time...")
        try:
            # Stream the pull progress
            current_digest = ""
            for progress in ollama.pull(self.model_name, stream=True):
                digest = progress.get("digest", "")
                if digest != current_digest and digest != "":
                    logger.info(f"Pulling layer {digest}...")
                    current_digest = digest

                status = progress.get("status", "")
                if status:
                     # Basic progress update, avoid excessive logging
                     if "pulling" in status or "downloading" in status:
                          if progress.get("completed") and progress.get("total"):
                                percent = round(progress['completed'] / progress['total'] * 100, 1)
                                print(f"\rStatus: {status} - {percent}%", end="")
                          else:
                                print(f"\rStatus: {status}...", end="")
                     else:
                          print(f"\rStatus: {status}") # Print final status for layer/pull

            print() # Newline after progress updates
            logger.info(f"Model '{self.model_name}' pulled successfully.")
            # Verify model availability after pull
            ollama.generate(model=self.model_name, prompt=".")
            logger.info(f"Model '{self.model_name}' confirmed available after pull.")

        except ollama.ResponseError as ep:
            logger.error(f'Error pulling model "{self.model_name}": {ep.error}')
            raise RuntimeError(f'Failed to pull Ollama model "{self.model_name}": {ep.error}') from ep
        except Exception as e:
            logger.error(f'Unexpected error pulling model "{self.model_name}": {e}', exc_info=True)
            raise RuntimeError(f'Unexpected error pulling Ollama model "{self.model_name}".') from e


    def _initialize_memory(self):
        """Initializes the conversation memory with file-based history."""
        history_file_path = self._get_history_file_path(self.session_id)
        logger.info(f"Initializing conversation memory with history file: {history_file_path}")
        logger.info(f"Memory summarization trigger token limit: {self.max_token_limit}")
        file_history = FileChatMessageHistory(history_file_path)
        # Ensure the directory exists for the history file
        os.makedirs(os.path.dirname(history_file_path), exist_ok=True)

        return ConversationSummaryBufferMemory(
            llm=self.llm,
            chat_memory=file_history,
            max_token_limit=self.max_token_limit, # Rely on this for summarization
            return_messages=True,
            memory_key="history", # Ensure memory key matches placeholder name
            input_key="input"     # Ensure input key matches placeholder name
        )

    def _get_history_file_path(self, session_id: str) -> str:
        """Constructs the full path for the chat history file."""
        # Ensure the base history directory exists
        os.makedirs(self.history_dir, exist_ok=True)
        return os.path.join(self.history_dir, f"{session_id}.json")

    def _initialize_chain(self):
        """Initializes the Langchain Runnable sequence."""
        logger.info("Setting up Langchain prompt and runnable...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"), # Matches memory_key
            ("human", "{input}"), # Matches input_key
        ])

        # Basic chain: prompt -> llm -> output parser
        base_chain = prompt | self.llm | StrOutputParser()

        # Wrap the base chain with message history management
        chain_with_history = RunnableWithMessageHistory(
            base_chain,
            # Use the already initialized memory object's chat_memory
            lambda session_id: self.memory.chat_memory,
            input_messages_key="input", # Key for user input in invoke/stream
            history_messages_key="history", # Key for history messages in the prompt
        )
        logger.info("Langchain runnable setup complete.")
        return chain_with_history

    # Removed custom summarize_messages method - ConversationSummaryBufferMemory handles it.

    def generate(self, input_text: str) -> str:
        """Generates a single response from the AI."""
        if not self.chain_with_message_history:
             logger.error("Cannot generate response: AI chain not initialized.")
             return "[ERROR: AI Chain not ready]"
        logger.debug(f"Generating response for input: '{input_text[:50]}...'")
        response = self.chain_with_message_history.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": self.session_id}}
        )
        if self.debug:
             logger.debug(f"Generated response: '{response[:100]}...'")
             # Log memory state if needed for debugging
             # logger.debug(f"Memory state: {self.memory.load_memory_variables({})}")
        return response

    def stream(self, input_text: str) -> Generator[str, None, None]:
        """Streams the AI's response chunk by chunk."""
        if not self.chain_with_message_history:
             logger.error("Cannot stream response: AI chain not initialized.")
             yield "[ERROR: AI Chain not ready]"
             return # Stop the generator

        logger.debug(f"Streaming response for input: '{input_text[:50]}...'")
        try:
            for chunk in self.chain_with_message_history.stream(
                {"input": input_text},
                config={"configurable": {"session_id": self.session_id}}
            ):
                yield chunk
        except Exception as e:
             logger.error(f"Error during AI model streaming: {e}", exc_info=True)
             yield "[ERROR: Streaming failed]"
        finally:
             if self.debug:
                  logger.debug("Streaming finished.")
                  # Log memory state after streaming if needed
                  # logger.debug(f"Memory state after stream: {self.memory.load_memory_variables({})}")

    # yield_generate removed, use stream directly

# Example Usage (Optional)
if __name__ == '__main__':
    lc.configure_logger(__name__, log_level=logging.DEBUG) # Enable debug for testing
    try:
        ai = AIModel(debug=True) # Enable debug mode for detailed Langchain logs
        print("AI Model Initialized.")
        print("Enter your message (or 'quit' to exit):")

        while True:
            user_input = input("YOU: ")
            if user_input.lower() == 'quit':
                break
            if not user_input.strip():
                continue

            print("AI: ", end="", flush=True)
            full_response = ""
            for chunk in ai.stream(user_input):
                print(chunk, end="", flush=True)
                full_response += chunk
            print() # Newline after response

    except RuntimeError as e:
        print(f"\nERROR: {e}")
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        logger.exception("Unexpected error in AIModel example usage.")
    finally:
        print("AI Model example finished.")