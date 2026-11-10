import os
import sys
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from loguru import logger
from langchain_openai import ChatOpenAI
from langsmith import traceable,get_current_run_tree
from langchain_core.callbacks import BaseCallbackHandler
import threading

load_dotenv()

MODEL = os.getenv("model")
OPENROUTER_API_KEY = os.getenv("openrouter_api_key")
BASE_URL = os.getenv("base_url")

# OpenRouter pricing is model-dependent. Configure these values in USD per
# 1 million tokens for the exact model configured in the environment.
INPUT_PRICE_PER_1M = float(os.getenv("LLM_INPUT_PRICE_PER_1M_USD", "0.10"))
OUTPUT_PRICE_PER_1M = float(os.getenv("LLM_OUTPUT_PRICE_PER_1M_USD", "0.32"))

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_PATH = PROJECT_ROOT / "vectordb"

LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
PROTOCOL_DOCUMENT_PATH = DATA_DIR
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

class LLMUsageTracker(BaseCallbackHandler):
    """
    Tracks actual token usage and estimated cost
    for LangChain LLM invocations.
    """

    def __init__(
        self,
        model: str,
        input_price_per_1m: float,
        output_price_per_1m: float,
    ):
        super().__init__()

        self.model = model
        self.input_price_per_1m = input_price_per_1m
        self.output_price_per_1m = output_price_per_1m

        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_call_count = 0
        self.usage_reported_call_count = 0

        self._lock = threading.Lock()

    def on_llm_end(
        self,
        response,
        **kwargs
    ) -> None:
        """
        Called when a LangChain model invocation completes.
        """

        usage = self._extract_usage(response)

        with self._lock:

            self.llm_call_count += 1

            if usage is None:
                logger.warning(
                    "LLM invocation completed without "
                    "reported token usage."
                )
                return

            input_tokens, output_tokens = usage

            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

            self.usage_reported_call_count += 1

            logger.info(
                f"LLM usage recorded | "
                f"model={self.model} | "
                f"input_tokens={input_tokens} | "
                f"output_tokens={output_tokens} | "
                f"total_tokens={input_tokens + output_tokens}"
            )

    @staticmethod
    def _extract_usage(response: Any) -> Optional[tuple]:

        if response is None:
            return None

        # First preference: provider token usage.
        llm_output = getattr(
            response,
            "llm_output",
            None
        ) or {}

        token_usage = llm_output.get(
            "token_usage",
            {}
        ) if isinstance(llm_output, dict) else {}

        if token_usage:

            input_tokens = token_usage.get(
                "prompt_tokens",
                token_usage.get("input_tokens")
            )

            output_tokens = token_usage.get(
                "completion_tokens",
                token_usage.get("output_tokens")
            )

            if input_tokens is not None and output_tokens is not None:
                return int(input_tokens), int(output_tokens)

        # Second preference: AIMessage.usage_metadata.
        for generation_group in getattr(
            response, "generations", []
        ):

            for generation in generation_group:

                message = getattr(
                    generation,
                    "message",
                    None
                )

                if message is None:
                    continue

                usage_metadata = getattr(
                    message,
                    "usage_metadata",
                    None
                )

                if usage_metadata:

                    return (
                        int(usage_metadata.get("input_tokens", 0)),
                        int(usage_metadata.get("output_tokens", 0))
                    )

                # Third preference: response_metadata.
                metadata = getattr(
                    message,
                    "response_metadata",
                    {}
                ) or {}

                usage = metadata.get(
                    "token_usage",
                    {}
                )

                if usage:

                    input_tokens = usage.get(
                        "prompt_tokens",
                        usage.get("input_tokens")
                    )

                    output_tokens = usage.get(
                        "completion_tokens",
                        usage.get("output_tokens")
                    )

                    if input_tokens is not None and output_tokens is not None:
                        return (
                            int(input_tokens),
                            int(output_tokens)
                        )

        return None

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    @property
    def input_cost_usd(self):
        return (
            self.input_tokens *
            self.input_price_per_1m / 1_000_000
        )

    @property
    def output_cost_usd(self):
        return (
            self.output_tokens *
            self.output_price_per_1m / 1_000_000
        )

    @property
    def total_cost_usd(self):
        return self.input_cost_usd + self.output_cost_usd

    def as_dict(self):

        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost_usd": self.input_cost_usd,
            "output_cost_usd": self.output_cost_usd,
            "total_cost_usd": self.total_cost_usd,
            "llm_call_count": self.llm_call_count,
            "usage_reported_call_count": (
                self.usage_reported_call_count
            ),
            "source": "langchain_callback_usage"
        }




class Settings:
    log_dir: str = str(LOG_DIR)
    model: str = MODEL
    vector_db_path: str = VECTOR_DB_PATH
    protocol_document_path: str  = PROTOCOL_DOCUMENT_PATH
    embedding_model: str = EMBEDDING_MODEL
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    top_k: int = TOP_K

    def get_logger(self, name: str = __name__):
        return logger.bind(module=name)

    @staticmethod
    def get_model_client():

        if not MODEL:
            raise ValueError("Model is not configured")

        if not OPENROUTER_API_KEY:
            raise ValueError("API key is not configured")

        client = ChatOpenAI(
            model=MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=BASE_URL,
            temperature=0,
            max_tokens=int(
                os.getenv("LLM_MAX_TOKENS", "1200")
            ),

            # OpenRouter usage reporting.
            extra_body={
                "usage": {
                    "include": True
                }
            },

            # Required when using streaming.
            stream_usage=True,

            max_retries=2,
            timeout=120,
        )

        logger.success(
            f"LangChain ChatOpenAI initialized | "
            f"model={MODEL}"
        )

        return client


class LoggerManager:
    @staticmethod
    def info(message: str):
        logger.info(message)

    @staticmethod
    def warning(message: str):
        logger.warning(message)

    @staticmethod
    def error(message: str):
        logger.error(message)

    @staticmethod
    def debug(message: str):
        logger.debug(message)

    @staticmethod
    def critical(message: str):
        logger.critical(message)


logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    colorize=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
)
logger.add(
    f"{LOG_DIR}/application.log",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    level="INFO",
    enqueue=True,
    backtrace=True,
    diagnose=True,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
)
logger.add(
    f"{LOG_DIR}/error.log",
    rotation="5 MB",
    retention="60 days",
    compression="zip",
    level="ERROR",
    enqueue=True,
    backtrace=True,
    diagnose=True,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
)

settings = Settings()