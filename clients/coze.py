import httpx
import os
import asyncio
from dotenv import load_dotenv
import json
import subprocess
import shlex


# Load environment variables from .env file
load_dotenv()

COZE_ENDPOINT = os.getenv("COZE_ENDPOINT", "")
COZE_API_KEY = os.getenv("COZE_API_KEY", "")
COZE_BOT_ID = os.getenv("COZE_BOT_ID", "")


def call_greco(sentences: str, quiet: bool = False) -> str:
    base_command = ["python3", "-m", "systems.greco2"]
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


class AsyncCoze:
    def __init__(self, api_key: str, timeout=30.0):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=timeout)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Connection": "keep-alive",
        }
        self.endpoint = COZE_ENDPOINT
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, outer):
            self.completions = self.Completions(outer)

        class Completions:
            def __init__(self, outer):
                self.outer = outer

            async def create(self, **model_params):
                # print(f"Sending API Request: {model_params}")
                # response = await self.outer.client.post(
                #     self.outer.endpoint,
                #     headers=self.outer.headers,
                #     json=model_params,
                # )
                # print(f"API Response Received: {response.json()}")

                print("coze query:", model_params["query"])

                # response = await call_greco_async(model_params["query"], True)
                response = await call_greco_async(model_params["query"], False)

                print("coze response:", response)

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
    client = AsyncCoze(api_key=COZE_API_KEY)
    # query = """I think that the public transport will always be in the future .
    # The rich people will buy a car but the poor people always need to use a bus or taxi .
    # I consider that is more convenient to drive a car because you carry on more things in your own car than travelling by car .
    # Also , you 'll meet friendly people who usually ask to you something to be friends and change your telephone number .
    # In my experience when I did n't have a car I used to use the bus to go to the school and go back to my house ."""
    query = "It 's difficult answer at the question \" what are you going to do in the future ? \" if the only one who has to know it is in two minds .\nWhen I was younger I used to say that I wanted to be a teacher , a saleswoman and even a butcher .. I do n't know why .\nI would like to study Psychology because one day I would open my own psychology office and help people .\nIt 's difficult because I 'll have to study hard and a lot , but I think that if you like a subject , you 'll study it easier .\nMaybe I 'll change my mind , maybe not .\nI think that the public transport will always be in the future .\nThe rich people will buy a car but the poor people always need to use a bus or taxi .\nI consider that is more convenient to drive a car because you carry on more things in your own car than travelling by car .\nAlso , you 'll meet friendly people who usually ask to you something to be friends and change your telephone number .\nIn my experience when I did n't have a car I used to use the bus to go to the school and go back to my house .\nIn my opinion , the car is n't necessary when you have crashed in the street , in that moment you realized the importance of a public transport .\nIn India we have various types of Public transport , like Cycle , Bike , Car , Train & Flight .\nDepending on the distance and duration to the desired place , mode of transport is chosen accordingly .\nBut Generally speaking , travelling by car is much more fun when compared with other modes of transport .\nThis reminds me of a trip that I have recently been to and the place is Agra .\nIt takes around 6 hours by National highway to go from Delhi to Agra .\nWe have stopped at hotels for having food and just in case if any of us feels hungry , we have purchased some snacks just before the trip .\nSince , we have the option to wait anytime we want to when we travel by car ( which is impossible when travelling by train & Flight ) .\nIn addition to it , we can also take a comfortable short nap on the back seat and wake up fresh .\nDue to the above mentioned reasons , I am going to conclude that travelling by car is much more convenient .\nMy name is Sarah .\nI am 17 years old .\nI am looking forward to join you in this year summer camps .\nI love children , and I enjoy looking after them . also , I organized many sports activities before in my school .\nIn addition to that , i enjoy cooking .\nMy family think that my cook is amazing .\nI hope that you give my the chance to join you .\nThanks\nMy favourite sport is volleyball because I love plays with my friends ."
    # query = (
    #     "This first sentence is.\nSecond is sentence.\nThird the is sentence."
    # )

    model_params = {
        "bot_id": COZE_BOT_ID,
        "user": "KyleToh",
        "query": query,
        "stream": False,
    }
    completion = await client.chat.completions.create(**model_params)
    response = completion.choices[0].message.content
    print(f"Final Extracted Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
