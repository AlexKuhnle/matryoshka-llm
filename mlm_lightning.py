import lightning
import tokenizers
import torch
from typing import Callable, Optional

from lm_lightning import LMLightning


class MLMLightning(lightning.LightningModule):

    @classmethod
    def get_non_matryoshka_module(cls):
        return LMLightning

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

    def get_nested_model(self, index):
        lightning_module = self.__class__
        model_module = self.model.__class__
        if index == 0:
            lightning_module = lightning_module.get_non_matryoshka_module()
            model_module = model_module.get_non_matryoshka_module()
        model_kwargs = self.model.get_nested_kwargs(index)
        nested_model = lightning_module(
            tokenizer=self.tokenizer,
            model=model_module,
            model_kwargs=model_kwargs,
            optimizer=self.optimizer_module,
            optimizer_kwargs=self.optimizer_kwargs,
            trainer_kwargs=self.trainer_kwargs,
        )
        nested_model.to(self.device)
        with torch.no_grad():
            self.model.init_nested_module(index, nested_model.model)
        return nested_model

    def forward(
        self,
        x="",
        num_outputs=1,
        index=-1,
        max_tokens=100,
        use_kv_cache=False,
    ):
        if num_outputs > 1:
            assert isinstance(x, str)
            x = torch.as_tensor(
                [self.tokenizer.encode(x).ids[-self.model.context_length - 1: -1]],
                dtype=torch.int64,
            ).repeat(num_outputs, 1).cuda()
        elif isinstance(x, str):
            x = torch.as_tensor(
                [self.tokenizer.encode(x).ids[-self.model.context_length - 1: -1]],
                dtype=torch.int64,
            ).cuda()
        else:
            raise NotImplementedError

        kv_cache = None
        if use_kv_cache:
            kv_cache = self.model.empty_kv_cache(x.size(0))

        x = MLMLightning._forward_iterative(self.eos_token, self.model, x, index, max_tokens, kv_cache)

        del kv_cache

        if num_outputs == 1:
            return self.tokenizer.decode(x[0].tolist())
        else:
            return [self.tokenizer.decode(seq.tolist()) for seq in x]

    @staticmethod
    def _forward_iterative(eos_token, model, x, index, max_tokens, kv_cache):
        eos = torch.zeros(size=[x.size(0), 1], dtype=torch.bool).cuda()

        for _ in range(max_tokens):
            if kv_cache is None or kv_cache[0][0].size(2) == 0:
                logits = model(x[:, -model.context_length:], kv_cache=kv_cache)[index][:, -1:]
            else:
                logits = model(x[:, kv_cache[0][0].size(2):], kv_cache=kv_cache)[index][:, -1:]

            next_token = logits.argmax(2)
            next_token = torch.where(eos, eos_token, next_token)
            x = torch.cat([x, next_token], dim=1)
            eos = torch.logical_or(eos, next_token == eos_token)
            if eos.all().item():
                return x

        return x

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
