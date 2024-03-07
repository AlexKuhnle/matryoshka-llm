import datasets
import lightning
import os
import sys
import tokenizers
import torch

from mlm_lightning import MLMLightning
from modules.mgpt_ablation import MGPTAblation
from modules.rms_norm import RMSNorm


if __name__ == "__main__":
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

    tokenizer = tokenizers.Tokenizer.from_file(f"data/tokenizers/{dataset_name}-bpe")
    os.makedirs(f"lightning_logs_lm/{dataset_name}-{model_name}-{suffix}", exist_ok=True)
    tokenizer.save(f"lightning_logs_lm/{dataset_name}-{model_name}-{suffix}/tokenizer")

    if model_name == "gpt":
        model_cls = MGPTAblation
    else:
        raise NotImplementedError

    model = MLMLightning(
        tokenizer=tokenizer,
        model=model_cls,
        model_kwargs=dict(
            vocab_size=tokenizer.get_vocab_size(),
            context_length=1024,
            prediction_sizes=[64, 128, 256, 512, 1024],
            prediction_multihead=True,
            num_trafos=8,
            trafo_size=1024,
            embedding_norm=False,
            position_scheme="rope",
            position_per_layer=True,
            normalization_module=RMSNorm,
            mhsa_num_heads=8,
            mhsa_kv_groups=None,
            mhsa_head_size=None,
            mhsa_qk_size=None,
            mhsa_torch_sdpa=True,
            mlp_hidden_sizes=[4096],
            mlp_activation_module=torch.nn.SiLU,
            mlp_glu=True,
            bias=False,
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
            batch_size=8,
            gradient_clipping=1.0,
            accumulate_grad_batches=1,
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

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=model.trainer_kwargs["batch_size"], num_workers=3,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=model.trainer_kwargs["batch_size"], num_workers=2,
    )

    logger = lightning.pytorch.loggers.TensorBoardLogger(
        "lightning_logs_lm", name=f"{dataset_name}-{model_name}-{suffix}",
    )

    trainer = lightning.Trainer(
        logger=logger,
        max_epochs=int(num_epochs),
        val_check_interval=0.05,
        gradient_clip_val=model.trainer_kwargs["gradient_clipping"],
        log_every_n_steps=1000,
        accumulate_grad_batches=model.trainer_kwargs["accumulate_grad_batches"],
    )

    trainer.fit(model, train_dataloader, test_dataloader)
