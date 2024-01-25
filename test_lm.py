import os
import sys
import tokenizers

from modules.gpt import GPT
from lm_lightning import LMLightning


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    suffix = sys.argv[3]

    lightning_directory = f"lightning_logs_lm/{dataset_name}-{model_name}-{suffix}"
    tokenizer = tokenizers.Tokenizer.from_file(f"{lightning_directory}/tokenizer")

    if model_name == "gpt":
        model_cls = GPT
    else:
        raise NotImplementedError

    version = sorted(
        int(x[8:]) for x in os.listdir(lightning_directory)
        if x.startswith("version_")
    )[-1]

    latest_epoch = -1
    latest_checkpoint = None
    for checkpoint in os.listdir(f"{lightning_directory}/version_{version}/checkpoints"):
        assert checkpoint.startswith("epoch=")
        epoch = int(checkpoint[checkpoint.index("=") + 1: checkpoint.index("-step")])
        assert epoch != latest_epoch
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_checkpoint = checkpoint
    assert latest_checkpoint is not None

    model = LMLightning.load_from_checkpoint(
        f"{lightning_directory}/version_{version}/checkpoints/{latest_checkpoint}",
        tokenizer=tokenizer,
        model=model_cls,
        model_kwargs=dict(vocab_size=tokenizer.get_vocab_size()),
        learning_rate=1e-4,
    )
    model.cuda()

    print(model("On a beautiful day", max_tokens=1000))
    for output in model(num_outputs=3, max_tokens=1000):
        print()
        print(output)
