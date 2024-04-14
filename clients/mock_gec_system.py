import httpx
import os
import asyncio
from dotenv import load_dotenv
import json
import subprocess
import shlex
import csv


# Load environment variables from .env file
load_dotenv()

COZE_ENDPOINT = os.getenv("COZE_ENDPOINT", "")
COZE_API_KEY = os.getenv("COZE_API_KEY", "")
COZE_BOT_ID = os.getenv("COZE_BOT_ID", "")


def call_greco(sentences: str, quiet: bool = False) -> str:
    base_command = ["python3", "-m", "systems.greco"]
    if quiet:
        base_command.append("--quiet")

    command = base_command + [shlex.quote(sentences)]

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Here you can log the error, raise a custom exception, or handle it in another way
        print(f"Error running command: {e}")
        raise


async def call_greco_async(sentences: str, quiet: bool = False) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, call_greco, sentences, quiet  # Uses default executor
    )


class Message:
    def __init__(self, content):
        self.content = content


class Choice:
    def __init__(self, message):
        self.message = message


class Completion:
    def __init__(self, choices):
        self.choices = choices

    @staticmethod
    def from_response(response):
        # print(f"Transforming response: {response}")
        primary_response = next(
            (
                msg
                for msg in response.get("messages", [])
                if msg.get("type") == "answer"
            ),
            None,
        )
        content = (
            primary_response.get("content", "No answer found in response.")
            if primary_response
            else "No answer found in response."
        )
        return Completion([Choice(Message(content))])


# TODO: change name to greco
class AsyncMockGECSystem:
    def __init__(self, csv_path: str):
        self.corrections = {}
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                original, corrected = row
                self.corrections[original] = corrected

    class Chat:
        def __init__(self, outer):
            self.completions = self.Completions(outer)
            self.outer = outer

        class Completions:
            def __init__(self, outer):
                self.outer = outer

            async def create(self, **model_params):
                original_sentences = json.loads(
                    model_params["messages"][1]["content"]
                )["input"].split("~~~")
                corrected_sentences = [
                    self.outer.corrections.get(sentence, sentence)
                    for sentence in original_sentences
                ]

                response = "~~~".join(corrected_sentences)

                response_json = {
                    "messages": [
                        {
                            "role": "assistant",
                            "type": "answer",
                            "content": json.dumps({"text": response}),
                            "content_type": "text",
                        }
                    ],
                    "conversation_id": "ac6bbded96a842d495953504a2721cb3",
                    "code": 0,
                    "msg": "success",
                }

                return Completion.from_response(response_json)


async def main():
    mock_gec_system = AsyncMockGECSystem(csv_path="path/to/your/csv.csv")
    chat = mock_gec_system.Chat(mock_gec_system)
    model_params = {
        "messages": [
            {"role": "system", "content": "Correct the following sentences:"},
            {
                "role": "user",
                "content": json.dumps(
                    {"input": "I goes to the park.~~~She eat an apple."}
                ),
            },
        ]
    }
    completion = await chat.completions.create(**model_params)
    for choice in completion.choices:
        print(choice.message.content)  # Print corrected sentences


if __name__ == "__main__":
    asyncio.run(main())
