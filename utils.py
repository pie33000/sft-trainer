import tiktoken
from tiktoken.core import Encoding


def get_tokenizer() -> Encoding:
    enc = tiktoken.get_encoding("gpt2")

    enc = tiktoken.Encoding(
        # If you're changing the set of special tokens, make sure to use a different name
        # It should be clear from the name what behaviour to expect.
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
