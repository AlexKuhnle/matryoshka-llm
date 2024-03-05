import lightning
import tokenizers
import torch
from typing import Callable, Optional


class MLMLightning(lightning.LightningModule):

    def __init__(
        self,
        tokenizer: tokenizers.Tokenizer,
        model: Callable[..., torch.nn.Module],
        model_kwargs: dict,
        optimizer: Optional[Callable[..., torch.optim.Optimizer]] = None,
        optimizer_kwargs: Optional[dict] = None,
        trainer_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        if optimizer is not None:
            self.save_hyperparameters({"model_kwargs": model_kwargs})
            self.save_hyperparameters("optimizer")
            self.save_hyperparameters({"optimizer_kwargs": optimizer_kwargs})
            self.save_hyperparameters({"trainer_kwargs": trainer_kwargs})

        self.tokenizer = tokenizer
        self.eos_token = self.tokenizer.token_to_id("</s>")
        self.pad_token = self.tokenizer.get_vocab_size()
        self.model = model(**model_kwargs)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token)
        self.optimizer_module = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.trainer_kwargs = trainer_kwargs

    def training_step(self, batch, batch_idx):
        x = batch["text"]
        if isinstance(x, str):
            x = torch.as_tensor([self.tokenizer.encode(x).ids], dtype=torch.int64).cuda()
        else:
            x = [torch.as_tensor(self.tokenizer.encode(seq).ids, dtype=torch.int64) for seq in x]
            x = torch.nn.utils.rnn.pad_sequence(
                x, batch_first=True, padding_value=self.pad_token,
            ).cuda()

        target = x[:, 1: self.model.context_length + 1].clone()
        x = x[:, :min(self.model.context_length, x.size(1) - 1)]
        x[x == self.pad_token] = self.eos_token
        losses = list()
        for n, logits in enumerate(self.model(x)):
            logits = logits.transpose(-2, -1)
            loss = self.loss(logits, target)
            losses.append(loss)
            self.log(f"train_loss{n}", loss)
        loss = torch.stack(losses).sum()
        self.log(f"train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch["text"]
        if isinstance(x, str):
            x = torch.as_tensor([self.tokenizer.encode(x).ids], dtype=torch.int64).cuda()
        else:
            x = [torch.as_tensor(self.tokenizer.encode(seq).ids, dtype=torch.int64) for seq in x]
            x = torch.nn.utils.rnn.pad_sequence(
                x, batch_first=True, padding_value=self.pad_token,
            ).cuda()

        target = x[:, 1: self.model.context_length + 1].clone()
        x = x[:, :min(self.model.context_length, x.size(1) - 1)]
        x[x == self.pad_token] = self.eos_token
        losses = list()
        metrics = dict()
        for n, logits in enumerate(self.model(x)):
            logits = logits.transpose(-2, -1)
            loss = self.loss(logits, target)
            losses.append(loss)
            self.log(f"test_loss{n}", loss, batch_size=x.size(0))
            metrics[f"loss{n}"] = loss
            accuracy = (logits.argmax(1) == target).sum() / (target != self.pad_token).sum()
            self.log(f"accuracy{n}", accuracy, batch_size=x.size(0))
            metrics[f"accuracy{n}"] = accuracy
        loss = torch.stack(losses).sum()
        self.log("test_loss", loss, batch_size=x.size(0))
        metrics["loss"] = loss
        return metrics

    def configure_optimizers(self):
        return self.optimizer_module(self.parameters(), **self.optimizer_kwargs)
