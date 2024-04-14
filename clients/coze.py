import httpx
import os
import asyncio
from dotenv import load_dotenv
import datetime


# Load environment variables from .env file
load_dotenv()

COZE_ENDPOINT = os.getenv("COZE_ENDPOINT", "")
COZE_API_KEY = os.getenv("COZE_API_KEY", "")
COZE_BOT_ID = os.getenv("COZE_BOT_ID", "")


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


class AsyncGreco:
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
                response = await self.outer.client.post(
                    self.outer.endpoint,
                    headers=self.outer.headers,
                    json=model_params,
                )
                # print(f"API Response Received: {response.json()}")
                return Completion.from_response(response.json())


async def main():
    client = AsyncGreco(api_key=COZE_API_KEY)
    # query = "{\"input\": \"By the way , my favorite football team is Manchester United , they are brilliant , they have an amazing football players , and they are awesome .<|NEXT|>\\nMichael was a little kid when he had a dream that was : Be a super hero !<|NEXT|>\\nAfter many years he still dream to become a super hero .<|NEXT|>\\nHe enter the university of medicine because he thinks that this profession was the more similar to be a super hero .<|NEXT|>\\nPass some years of the university and he know a girl called Kate and he get loved on here and she get loved in him .<|NEXT|>\\nKate was cursing the university of fashion .<|NEXT|>\\nThen the two started to date .<|NEXT|>\\nMichael and Kate was so happy , until one day that Michael said to him dream to be a super hero , Kate get so nervous saying that it was ridiculous and just a kid dream and that Michael was only dreaming but that would never happen and that super hero do n't exist .<|NEXT|>\\nWhen she said that , Michael started to cry and get mad saying that she was lying and do n't have heart .<|NEXT|>\\nHe got so mad that he asked her to get out of his apartment .<|NEXT|>\\nIn following day , she said sorry and them was happy again .<|NEXT|>\\nBut when them get to the home of Michael they fight again and Michael get a knife and kill her .<|NEXT|>\\nHe cried with a lot blood around .<|NEXT|>\\nMichael get away from there .<|NEXT|>\\nMichael closed the door and knew at that moment he had made a mistake .<|NEXT|>\\nPublic transportation is an important invention in human history , it brings amount of benefits in our life .<|NEXT|>\\nFor instance , we take subway in order to avoid stocking in traffic .<|NEXT|>\\nIt is true that moving by car is more convenient than take public transportation , however , it would cause more damage to our life and harm our environment .<|NEXT|>\\nTherefore , we need to think more about our future , our offsprings .<|NEXT|>\\nGiving them a safe , clean and comfortable place to live .<|NEXT|>\\nIf there is no public transportation , traffic jam will be serious than before .<|NEXT|>\\nIn my country , taipei , we always stock in traffic for about one hour in the morning .<|NEXT|>\\nI can not imagine if there is no bus or MRT , how long I will take for school .<|NEXT|>\\nCar is convenient for human , but it brings damage for human also .<|NEXT|>\\nFor example , carbon dioxide which is created by car .<|NEXT|>\\nIt cause global warming which threats our environment and harms our daily life .<|NEXT|>\\nwe can see that there are lots of serious and frequently weather disaster happened in decades , such as typhoon , hurricane , wild fire and mud slide .<|NEXT|>\\nWhat other precautions , is usually taken ?<|NEXT|>\\nThe NG Office is notified by the photographer when the film was shipped .<|NEXT|>\\nIf the film does n't arrive on time , it immediately .<|NEXT|>\\nLost shipments have been found more easily when this process will be started right away<|NEXT|>\\nThe pleasure of traveling<|NEXT|>\\nPeople go , people come .<|NEXT|>\\nEvery day lots of people are travelling abroad but , what 's the best way to do an international travel ?<|NEXT|>\\nThe most important option to travel is , by far , the plane .\"}"

    query = """{"original_sentence": ["It 's difficult answer at the question \" what are you going to do in the future ? \" if the only one who has to know it is in two minds .", "When I was younger I used to say that I wanted to be a teacher , a saleswoman and even a butcher .. I do n't know why .", "I would like to study Psychology because one day I would open my own psychology office and help people .", "It 's difficult because I 'll have to study hard and a lot , but I think that if you like a subject , you 'll study it easier .", "Maybe I 'll change my mind , maybe not .", "I think that the public transport will always be in the future .", "The rich people will buy a car but the poor people always need to use a bus or taxi .", "I consider that is more convenient to drive a car because you carry on more things in your own car than travelling by car .", "Also , you 'll meet friendly people who usually ask to you something to be friends and change your telephone number .", "In my experience when I did n't have a car I used to use the bus to go to the school and go back to my house .", "In my opinion , the car is n't necessary when you have crashed in the street , in that moment you realized the importance of a public transport .", "In India we have various types of Public transport , like Cycle , Bike , Car , Train & Flight .", "Depending on the distance and duration to the desired place , mode of transport is chosen accordingly .", "But Generally speaking , travelling by car is much more fun when compared with other modes of transport .", "This reminds me of a trip that I have recently been to and the place is Agra .", "It takes around 6 hours by National highway to go from Delhi to Agra .", "We have stopped at hotels for having food and just in case if any of us feels hungry , we have purchased some snacks just before the trip .", "Since , we have the option to wait anytime we want to when we travel by car ( which is impossible when travelling by train & Flight ) .", "In addition to it , we can also take a comfortable short nap on the back seat and wake up fresh .", "Due to the above mentioned reasons , I am going to conclude that travelling by car is much more convenient .", "My name is Sarah .", "I am 17 years old .", "I am looking forward to join you in this year summer camps .", "I love children , and I enjoy looking after them . also , I organized many sports activities before in my school .", "In addition to that , i enjoy cooking .", "My family think that my cook is amazing .", "I hope that you give my the chance to join you .", "Thanks", "My favourite sport is volleyball because I love plays with my friends ."], "student_sentence": ["It 's difficult to answer the question \" what are you going to do in the future ? \" if the only one who has to know it is in two minds .", "When I was younger , I used to say that I wanted to be a teacher , a saleswoman and even a butcher . I do n't know why .", "I would like to study Psychology because one day I would open my own psychology office and help people .", "It 's difficult because I 'll have to study hard and a lot , but I think that if you like a subject , you 'll study it easier .", "Maybe I 'll change my mind , maybe not .", "I think that public transport will always be in the future .", "The rich people will buy a car , but poor people always need to use a bus or taxi .", "I consider that it is more convenient to drive a car because you carry on more things in your own car than travelling by car .", "Also , you 'll meet friendly people who usually ask you something to be friends and change your telephone number .", "In my experience , when I did n't have a car , I used to use the bus to go to school and go back to my house .", "In my opinion , the car is n't necessary when you have crashed in the street . In that moment you realized the importance of public transport .", "In India we have various types of public transport , like cycle , bike , car , train & flight .", "Depending on the distance and duration to the desired place , mode of transport is chosen accordingly .", "But Generally speaking , travelling by car is much more fun when compared with other modes of transport .", "This reminds me of a trip that I have recently been to and the place is Agra .", "It takes around 6 hours by National highway to go from Delhi to Agra .", "We have stopped at hotels for food and , just in case any of us feels hungry , we have purchased some snacks just before the trip .", "Since , we have the option to wait anytime we want to when we travel by car ( which is impossible when travelling by train & flight ) .", "In addition to it , we can also take a comfortable short nap in the back seat and wake up fresh .", "Due to the above mentioned reasons , I am going to conclude that travelling by car is much more convenient .", "My name is Sarah .", "I am 17 years old .", "I am looking forward to joining you in this year 's summer camps .", "I love children , and I enjoy looking after them . Also , I organized many sports activities before in my school .", "In addition to that , I enjoy cooking .", "My family think that my cook is amazing .", "I hope that you give me the chance to join you .", "Thanks", "My favourite sport is volleyball because I love playing with my friends ."]}"""

    model_params = {
        "bot_id": COZE_BOT_ID,
        "user": "KyleToh",
        "query": query,
        "stream": False,
    }
    start_time = datetime.datetime.now()  # Capture start time
    completion = await client.chat.completions.create(**model_params)
    end_time = datetime.datetime.now()  # Capture end time

    response = completion.choices[0].message.content
    print(f"Final Extracted Response: {response}")

    # Calculate and print the duration
    duration = end_time - start_time
    print(f"Time taken: {duration.total_seconds()} seconds")


if __name__ == "__main__":
    asyncio.run(main())
