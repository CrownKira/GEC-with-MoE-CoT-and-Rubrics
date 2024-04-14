import asyncio
import json
import csv


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
        self.chat = self.Chat(self)
        self.corrections = {}

        # TODO: refactor
        csv_path = "clients/mock_data/" + csv_path
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

                # TODO: throw error when cant get
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


# async def main():
#     mock_gec_system = AsyncMockGECSystem(
#         csv_path="ABCN.dev.gold.bea19.BART-A.corrected.csv"
#     )
#     chat = mock_gec_system.Chat(mock_gec_system)
#     model_params = {
#         "messages": [
#             {"role": "system", "content": "Correct the following sentences:"},
#             {
#                 "role": "user",
#                 "content": json.dumps(
#                     {
#                         "input": "Maybe I 'll change my mind , maybe not .~~~I think that the public transport will always be in the future ."
#                     }
#                 ),
#             },
#         ]
#     }
#     completion = await chat.completions.create(**model_params)
#     for choice in completion.choices:
#         print(choice.message.content)  # Print corrected sentences


# if __name__ == "__main__":
#     asyncio.run(main())
