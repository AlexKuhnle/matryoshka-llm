import lightning
import tokenizers
import torch
from typing import Callable, Optional


class LMLightning(lightning.LightningModule):

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

        def assign_log_function_to_module(module, log_prefix):
            def fn_log(name, value):
                if self.model.training:
                    self.log(f"{log_prefix}-{name}", value, batch_size=1)
            assert not hasattr(module, "log")
            setattr(module, "log", fn_log)

        for module_name, module in self.model.named_modules():
            assign_log_function_to_module(module, f"gpt-{module_name}")

    def forward(self, x="", num_outputs=1, max_tokens=100, use_kv_cache=True):
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

        batch_size = x.size(0)
        eos = torch.zeros(size=[batch_size], dtype=torch.bool).cuda()
        if use_kv_cache:
            kv_cache = self.model.empty_kv_cache(batch_size)

        for iteration in range(max_tokens):
            if use_kv_cache:
                if iteration == 0:
                    logits, kv_cache = self.model(x, kv_cache=kv_cache)
                else:
                    logits, kv_cache = self.model(x[:, -1:], kv_cache=kv_cache)
            else:
                logits = self.model(x)
            
            logits = logits[:, -1]
            next_token = logits.argmax(1)
            next_token = torch.where(eos, self.eos_token, next_token)
            x = torch.cat([x[-self.model.context_length + 1:], next_token.unsqueeze(1)], dim=1)
            eos = torch.logical_or(eos, next_token == self.eos_token)
            if eos.all().item():
                break

        if use_kv_cache:
            del kv_cache

        if num_outputs == 1:
            return self.tokenizer.decode(x[0].tolist())
        else:
            return [self.tokenizer.decode(seq.tolist()) for seq in x]

    def training_step(self, batch, batch_idx):
        x = batch["text"]
        if isinstance(x, str):
            x = torch.as_tensor([self.tokenizer.encode(x).ids], dtype=torch.int64).cuda()
        else:
            x = [torch.as_tensor(self.tokenizer.encode(seq).ids, dtype=torch.int64) for seq in x]
            x = torch.nn.utils.rnn.pad_sequence(
                x, batch_first=True, padding_value=self.pad_token,
            ).cuda()

        x, target = (
            x[:, :min(self.model.context_length, x.size(1) - 1)],
            x[:, 1: self.model.context_length + 1],
        )
        x[x == self.pad_token] = self.eos_token
        logits = self.model(x)
        logits = logits.transpose(-2, -1)
        loss = self.loss(logits, target)
        self.log("train_loss", loss)
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

        x, target = (
            x[:, :min(self.model.context_length, x.size(1) - 1)],
            x[:, 1: self.model.context_length + 1],
        )
        x[x == self.pad_token] = self.eos_token
        logits = self.model(x)
        logits = logits.transpose(-2, -1)
        loss = self.loss(logits, target)
        self.log("test_loss", loss, batch_size=x.size(0))
        accuracy = (logits.argmax(1) == target).sum() / (target != self.pad_token).sum()
        self.log("accuracy", accuracy, batch_size=x.size(0))
        return {"loss": loss, "accuracy": accuracy}

    def configure_optimizers(self):
        return self.optimizer_module(self.parameters(), **self.optimizer_kwargs)
