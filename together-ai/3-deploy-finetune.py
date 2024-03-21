import together


# https://docs.together.ai/reference/instances-start

together.Models.start(
    "wi2404@gmail.com/RedPajama-INCITE-Chat-3B-v1-my-demo-finetune-2024-03-17-12-28-00"
)


output = together.Complete.create(
    prompt="Isaac Asimov's Three Laws of Robotics are:\n\n1. ",
    model="wi2404@gmail.com/RedPajama-INCITE-Chat-3B-v1-my-demo-finetune-2024-03-17-12-28-00",
)
