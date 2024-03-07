import datasets
import os
import sys
import timeit
import tokenizers
import torch
from tqdm.auto import tqdm
import yaml

from modules.mgpt import MGPT
from modules.mrms_norm import MRMSNorm
from mlm_lightning import MLMLightning


def load_model(dataset_name, model_name, suffix):
    lightning_directory = f"lightning_logs_lm/{dataset_name}-{model_name}-{suffix}"
    tokenizer = tokenizers.Tokenizer.from_file(f"{lightning_directory}/tokenizer")

    if model_name == "gpt":
        model_cls = MGPT
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

    with open(f"{lightning_directory}/version_{version}/hparams.yaml") as file:
        model_kwargs = yaml.full_load(file)["model_kwargs"]
    model = MLMLightning.load_from_checkpoint(
        f"{lightning_directory}/version_{version}/checkpoints/{latest_checkpoint}",
        tokenizer=tokenizer,
        model=model_cls,
        model_kwargs=model_kwargs,
    )
    model.cuda()
    model.eval()  # TODO: why not default?

    return model


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    suffix = sys.argv[3]

    if dataset_name == "tinystories":
        dataset = "roneneldan/TinyStories"
    elif dataset_name == "tinyshakespeare":
        dataset = "tiny_shakespeare"
    else:
        raise NotImplementedError

    dataset = datasets.load_dataset(dataset)
    test_dataset = dataset["validation"]

    model = load_model(dataset_name, model_name, suffix)

    losses, accuracies = model.evaluate(tqdm(test_dataset))
    for index, (loss, accuracy) in enumerate(zip(losses, accuracies)):
        print(f"{index} {model.model.prediction_sizes[index]}: {loss:.4f} {(accuracy * 100):.1f}%")
