from openai import OpenAI

client = OpenAI()

# https://platform.openai.com/storage

print(
    client.files.create(
        file=open("demo-train.jsonl", "rb"), purpose="fine-tune"
    )
)

print(
    client.files.create(
        file=open("demo-validate.jsonl", "rb"), purpose="fine-tune"
    )
)
