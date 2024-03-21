import together
import os


# Uploading antihallucination.jsonl: 100%|████████████████████████████████████████████████████████████████████| 763k/763k [00:03<00:00, 218kB/s]
# {
#     "filename": "antihallucination.jsonl",
#     "id": "file-7583d942-59b6-45ae-b26e-23c73963bdbf",
#     "object": "file",
#     "report_dict": {
#         "is_check_passed": true,
#         "model_special_tokens": "we are not yet checking end of sentence tokens for this model",
#         "file_present": "File found",
#         "file_size": "File size 0.001 GB",
#         "num_samples": 238
#     }
# }


resp = together.Finetune.create(
    training_file="file-7583d942-59b6-45ae-b26e-23c73963bdbf",
    model="togethercomputer/RedPajama-INCITE-Chat-3B-v1",
    n_epochs=3,
    n_checkpoints=1,
    batch_size=4,
    learning_rate=1e-5,
    suffix="my-demo-finetune",
    wandb_api_key=os.getenv("WANDB_API_KEY", ""),
)


fine_tune_id = resp["id"]
print(resp)


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
