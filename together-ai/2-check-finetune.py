import together

# https://api.together.xyz/jobs

# fine_tune_id = "ft-61aadb76-c156-42b7-a508-207df88ea3cc"
fine_tune_id = "ft-31eebad2-04fd-4d9a-9a86-20e2b40c5ef4"

print(together.Models.list())  # retrieves information on finetune event


print(
    together.Finetune.retrieve(fine_tune_id=fine_tune_id)
)  # retrieves information on finetune event
print(
    together.Finetune.get_job_status(fine_tune_id=fine_tune_id)
)  # pending, running, completed
print(
    together.Finetune.is_final_model_available(fine_tune_id=fine_tune_id)
)  # True, False
print(
    together.Finetune.get_checkpoints(fine_tune_id=fine_tune_id)
)  # list of checkpoints
