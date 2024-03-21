from typing import List
from tiktoken import get_encoding
import sys


# def count_tokens(text: str) -> int:
#     # Placeholder implementation
#     return len(text.split())


def count_tokens(text: str) -> int:
    enc = get_encoding("gpt2")
    tokens = enc.encode(text)
    token_count = len(tokens)
    return token_count


def split_text_into_batches(
    text: str,
    batch_size_in_tokens: int = 10,
) -> List[str]:
    lines = text.split("\n")
    batches = []
    current_batch = ""
    current_batch_tokens = 0

    index = 1

    for line in lines:
        line_tokens = count_tokens(line + "\n")
        print("Line", index, line_tokens, line)

        index += 1

        if line_tokens > batch_size_in_tokens:
            print(
                f"Error: Line exceeds the batch size of {batch_size_in_tokens} tokens."
            )
            print("Line:", line)
            print("Tokens:", line_tokens)
            sys.exit(1)

        if current_batch_tokens + line_tokens < batch_size_in_tokens:
            current_batch += line + "\n"
            current_batch_tokens += line_tokens
        else:
            batches.append(current_batch.strip())
            current_batch = line + "\n"
            current_batch_tokens = line_tokens

    if current_batch.strip():
        batches.append(current_batch.strip())

    return batches


# Test the split_text_into_batches function
sample_text = """
This is the first line.
This is the second line, which is longer.
Third line.
Fourth line, even longer than the second.
Fifth and final line.
"""

print("Original text:")
print(sample_text)

batch_size = 20
overlap = 2


batches = split_text_into_batches(
    sample_text,
    batch_size,
)

print(f"\nBatches (batch_size={batch_size}, overlap={overlap}):")
for i, batch in enumerate(batches, start=1):
    print(f"Batch {i}:")
    print(batch)
    print("Tokens:", count_tokens(batch))
    print()
