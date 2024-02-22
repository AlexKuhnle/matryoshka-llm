import datasets
import lightning
import os
import sys
import tokenizers
import torch

from modules.gpt import GPT
from modules.rms_norm import RMSNorm
from lm_lightning import LMLightning


if __name__ == "__main__":
    DEBUG = False
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    suffix = sys.argv[3]
    num_epochs = float(sys.argv[4])

    if dataset_name == "tinystories":
        dataset = "roneneldan/TinyStories"
    elif dataset_name == "tinyshakespeare":
        dataset = "tiny_shakespeare"
    else:
        raise NotImplementedError

    dataset = datasets.load_dataset(dataset)
    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]

    if num_epochs < 1.0:
        train_dataset = train_dataset.select(list(range(int(len(train_dataset) * num_epochs))))
        num_epochs = 1
    elif DEBUG:
        train_dataset = train_dataset.select(list(range(10)))
        test_dataset = test_dataset.select(list(range(10)))

    tokenizer = tokenizers.Tokenizer.from_file(f"data/tokenizers/{dataset_name}-bpe")
    os.makedirs(f"lightning_logs_lm/{dataset_name}-{model_name}-{suffix}", exist_ok=True)
    tokenizer.save(f"lightning_logs_lm/{dataset_name}-{model_name}-{suffix}/tokenizer")

    if model_name == "gpt":
        model_cls = GPT
    else:
        raise NotImplementedError

    model = LMLightning(
        tokenizer=tokenizer,
        model=model_cls,
        model_kwargs=dict(
            vocab_size=tokenizer.get_vocab_size(),
            context_length=1024,
            num_trafos=8,
            trafo_size=512,
            embedding_norm=True,
            position_scheme="rope",
            position_per_layer=True,
            normalization_module=RMSNorm,  # RMSNorm
            mhsa_num_heads=16,
            mhsa_kv_groups=None,
            mhsa_head_size=32,
            mhsa_qk_size=None,
            mhsa_torch_sdpa=True,
            mhsa_flash_sdpa=False,
            mlp_hidden_sizes=[512],  # * 4
            mlp_activation_module=torch.nn.SiLU,  # SiLU
            mlp_glu=True,  # True
            bias=True,
            dropout=0.0,
        ),
        optimizer=torch.optim.Adam,
        optimizer_kwargs=dict(
            lr=1e-3,
            # betas=(0.9, 0.95),
            # eps=1e-5,
            # weight_decay=0.1,
        ),
        trainer_kwargs=dict(
            batch_size=16,
            gradient_clipping=1.0,  # 1.0
        )
    )
    model.cuda()

    if dataset_name == "tinyshakespeare":

        def split_tinyshakespeare(dataset):
            text = dataset[0]["text"]
            while "\n\n\n" in text:
                text = text.replace("\n\n\n", "\n\n")
            paragraphs = list(text.split("\n\n"))
            merged = [paragraphs[0]]
            for paragraph in paragraphs[1:]:
                if len(merged[-1]) / 4 < model.model.context_length:
                    merged[-1] = merged[-1] + paragraph
                else:
                    merged.append(paragraph)
            return datasets.Dataset.from_dict({"text": merged})

        train_dataset = split_tinyshakespeare(train_dataset)
        test_dataset = split_tinyshakespeare(test_dataset)

    print(f"train: {len(train_dataset)}")
    print(f"test:  {len(test_dataset)}")
    print(f"vocab: {tokenizer.get_vocab_size()}")

    batch_size = 8
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=model.trainer_kwargs["batch_size"],
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=model.trainer_kwargs["batch_size"],
    )

    logger = lightning.pytorch.loggers.TensorBoardLogger(
        "lightning_logs_lm", name=f"{dataset_name}-{model_name}-{suffix}",
    )
    trainer = lightning.Trainer(
        logger=logger,
        max_epochs=int(num_epochs),
        # limit_val_batches=0.1,  # (1.0 if DEBUG else 0.01),
        val_check_interval=0.05,  # (1.0 if DEBUG else 0.01),
        gradient_clip_val=model.trainer_kwargs["gradient_clipping"],
    )

    trainer.fit(model, train_dataloader, test_dataloader)
