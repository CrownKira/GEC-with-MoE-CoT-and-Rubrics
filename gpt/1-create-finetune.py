from openai import OpenAI

client = OpenAI()

# https://platform.openai.com/finetune

# To set additional fine-tuning parameters like the validation_file or hyperparameters, please refer to the API specification for fine-tuning.
# gpt-3.5-turbo-0125 (recommended), gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, babbage-002, davinci-002, and gpt-4-0613 (experimental).
print(
    client.fine_tuning.jobs.create(
        training_file="file-Je2bfokrhe86EgTV3N09LmlC",
        model="gpt-3.5-turbo",
        validation_file="file-zy9BH51My4ML02AkbkJWH7EL",
    )
)
