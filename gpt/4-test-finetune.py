from openai import OpenAI

client = OpenAI()

# https://platform.openai.com/finetune
# ft:gpt-3.5-turbo-0125:personal::93m7VXbd

completion = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0125:personal::93m7VXbd",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)
print(completion.choices[0].message)
