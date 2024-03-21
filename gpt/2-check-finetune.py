from openai import OpenAI

client = OpenAI()

# https://platform.openai.com/finetune

# List 10 fine-tuning jobs
print(client.fine_tuning.jobs.list(limit=10))

# # Retrieve the state of a fine-tune
# client.fine_tuning.jobs.retrieve("ftjob-abc123")

# # Cancel a job
# client.fine_tuning.jobs.cancel("ftjob-abc123")

# # List up to 10 events from a fine-tuning job
# client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-abc123", limit=10)

# # Delete a fine-tuned model (must be an owner of the org the model was created in)
# client.models.delete("ft:gpt-3.5-turbo:acemeco:suffix:abc123")
