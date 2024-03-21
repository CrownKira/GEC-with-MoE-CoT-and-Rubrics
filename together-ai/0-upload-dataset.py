import together
import os

together.api_key = os.getenv("TOGETHER_API_KEY", "")


# check prompt format
# together models info togethercomputer/llama-2-7b-chat


# resp = together.Files.check(file="antihallucination.jsonl")
# print(resp)


resp = together.Files.upload(file="antihallucination.jsonl")
print(resp)


# together files upload antihallucination.jsonl
