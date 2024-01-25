import datasets
import lightning
import os
import sys
import tokenizers
import torch

from modules.gpt import GPT
from lm_lightning import LMLightning


if __name__ == "__main__":
    DEBUG = False
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    suffix = sys.argv[3]
    num_epochs = float(sys.argv[4])

    if dataset_name == "tinystories":
        dataset = "roneneldan/TinyStories"
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

    print(f"train: {len(train_dataset)}")
    print(f"test:  {len(test_dataset)}")

    batch_size = 8
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    tokenizer = tokenizers.Tokenizer.from_file(f"data/tokenizers/{dataset_name}-bpe")
    os.makedirs(f"lightning_logs_lm/{dataset_name}-{model_name}-{suffix}", exist_ok=True)
    tokenizer.save(f"lightning_logs_lm/{dataset_name}-{model_name}-{suffix}/tokenizer")
    print(f"vocab: {tokenizer.get_vocab_size()}")

    if model_name == "gpt":
        model_cls = GPT
    else:
        raise NotImplementedError

    model = LMLightning(
        tokenizer=tokenizer,
        model=model_cls,
        model_kwargs=dict(vocab_size=tokenizer.get_vocab_size()),
        learning_rate=1e-4,
    )
    model.cuda()

    logger = lightning.pytorch.loggers.TensorBoardLogger(
        "lightning_logs_lm", name=f"{dataset_name}-{model_name}-{suffix}",
    )
    trainer = lightning.Trainer(
        logger=logger,
        max_epochs=int(num_epochs),
        limit_val_batches=(1.0 if DEBUG else 0.01),
        val_check_interval=(1.0 if DEBUG else 0.01),
    )

    trainer.fit(model, train_dataloader, test_dataloader)
