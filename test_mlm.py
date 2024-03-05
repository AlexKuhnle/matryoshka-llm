import os
import sys
import timeit
import tokenizers
import torch
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
    # model.cuda()
    model.eval()  # TODO: why not default?

    return model


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    suffix = sys.argv[3]

    model = load_model(dataset_name, model_name, suffix)

    # if len(sys.argv) == 5:
    #     speculative_suffix = sys.argv[4]
    #     speculative_model = load_model(dataset_name, model_name, speculative_suffix)
    #     speculative_model = speculative_model.model  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # else:
    speculative_model = None

    with torch.inference_mode():

        print(model("On a beautiful day", max_tokens=1000, speculative_model=speculative_model, speculative_horizon=8))
        print()
        print(model(max_tokens=1000, speculative_model=speculative_model, speculative_horizon=8))
        print()

        target = model(max_tokens=1000)
        assert model(max_tokens=1000, use_kv_cache=False) == target
        assert all(output == target for output in model(num_outputs=5, max_tokens=1000))
        assert model(max_tokens=1000, speculative_model=model.model, speculative_horizon=8) == target
        assert model(max_tokens=1000, speculative_model=speculative_model, speculative_horizon=8) == target
        assert model(max_tokens=1000, use_kv_cache=False, speculative_model=speculative_model, speculative_horizon=8) == target

        prompt = target[: len(target) // 2]
        print("kv-cache  prompt  speculative  horizon  ")
        print("   -        -          -          -     ",
            timeit.timeit((lambda: model(max_tokens=1000, use_kv_cache=False)), number=100)
        )
