import logging

import tiktoken
from tiktoken.core import Encoding


def get_tokenizer(model_name: str) -> Encoding:
    enc = tiktoken.get_encoding(model_name)

    enc = tiktoken.Encoding(
        name="enc_sft",
        pat_str=enc._pat_str,
        mergeable_ranks=enc._mergeable_ranks,
        special_tokens={
            **enc._special_tokens,
            "<|endofprompt|>": 50257,
            "<|pad|>": 50258,
        },
    )
    return enc


def setup_logger(name, level=logging.INFO):
    """
    Sets up a logger with the specified name and log level.

    Parameters:
    - name (str): Name of the logger.
    - level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
    - logger (logging.Logger): Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if the logger already has handlers configured (to avoid duplicate messages)
    if not logger.hasHandlers():
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

    return logger
