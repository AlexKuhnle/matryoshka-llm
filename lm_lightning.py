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

    def configure_optimizers(self):
        return self.optimizer_module(self.parameters(), **self.optimizer_kwargs)

    def forward(
        self,
        x="",
        num_outputs=1,
        max_tokens=100,
        use_kv_cache=False,
        speculative_model=None,
        speculative_horizon=7,
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
        speculative_kv_cache = None
        if use_kv_cache:
            kv_cache = self.model.empty_kv_cache(x.size(0))
            if speculative_model is not None:
                speculative_kv_cache = speculative_model.empty_kv_cache(x.size(0))

        if speculative_model is None:
            x = LMLightning._forward_iterative(self.eos_token, self.model, x, max_tokens, kv_cache)
        else:
            x = LMLightning._forward_speculative(
                self.eos_token, self.model, x, max_tokens, kv_cache,
                speculative_model, speculative_horizon, speculative_kv_cache
            )

        del kv_cache
        del speculative_kv_cache

        if num_outputs == 1:
            return self.tokenizer.decode(x[0].tolist())
        else:
            return [self.tokenizer.decode(seq.tolist()) for seq in x]

    @staticmethod
    def _forward_iterative(eos_token, model, x, max_tokens, kv_cache):
        eos = torch.zeros(size=[x.size(0), 1], dtype=torch.bool).cuda()

        for _ in range(max_tokens):
            if kv_cache is None or kv_cache[0][0].size(2) == 0:
                logits = model(x[:, -model.context_length:], kv_cache=kv_cache)[:, -1:]
            else:
                logits = model(x[:, kv_cache[0][0].size(2):], kv_cache=kv_cache)[:, -1:]

            next_token = logits.argmax(2)
            next_token = torch.where(eos, eos_token, next_token)
            x = torch.cat([x, next_token], dim=1)
            eos = torch.logical_or(eos, next_token == eos_token)
            if eos.all().item():
                return x

        return x

    @staticmethod
    def _forward_speculative(
        eos_token, model, x, max_tokens, kv_cache,
        speculative_model, speculative_horizon, speculative_kv_cache
    ):
        num_generated = 0
        while num_generated < max_tokens:
            num_tokens = x.size(1)
            if num_generated > 0 and kv_cache is not None:
                assert kv_cache[0][0].size(2) == num_tokens - 1
                assert speculative_kv_cache[0][0].size(2) == num_tokens - 2
            speculative = LMLightning._forward_iterative(
                eos_token, speculative_model, x, speculative_horizon, speculative_kv_cache
            )
            assert speculative.size(1) - num_tokens <= speculative_horizon

            if kv_cache is None or num_generated == 0:
                logits = model(speculative[:, -model.context_length:], kv_cache=kv_cache)[:, num_tokens - 1:]
            else:
                logits = model(speculative[:, num_tokens - 1:], kv_cache=kv_cache)

            next_tokens = logits.argmax(2)

            mismatch_indices = (speculative[:, num_tokens:] != next_tokens[:, :-1]).any(0).nonzero()
            any_mismatch = (mismatch_indices.numel() > 0)
            if any_mismatch:
                first_mismatch = mismatch_indices[0].item()
                x = speculative[:, :num_tokens + first_mismatch]
                if kv_cache is not None:
                    remove_cache = speculative.size(1) - num_tokens - first_mismatch
                    kv_cache = [
                        (c1[:, :, :-remove_cache], c2[:, :, :-remove_cache])
                        for c1, c2 in kv_cache
                    ]
                    speculative_kv_cache = [
                        (c1[:, :, :-remove_cache], c2[:, :, :-remove_cache])
                        for c1, c2 in speculative_kv_cache
                    ]
            else:
                first_mismatch = speculative.size(1) - num_tokens
                x = speculative

            eos = (x[:, -1:] == eos_token)
            if eos.all().item():
                return x

            next_token = next_tokens[:, first_mismatch: first_mismatch + 1]
            assert next_token.numel() > 0
            next_token = torch.where(eos, eos_token, next_token)
            x = torch.cat([x, next_token], dim=1)
            eos = torch.logical_or(eos, next_token == eos_token)
            if eos.all().item():
                return x

            num_generated += first_mismatch + 1

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
        logits = self.model(x)
        logits = logits.transpose(-2, -1)
        loss = self.loss(logits, target)
        self.log(f"train_loss{self.model.trafo_size}", loss)
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
        logits = self.model(x)
        logits = logits.transpose(-2, -1)
        loss = self.loss(logits, target)
        self.log(f"test_loss{self.model.trafo_size}", loss, batch_size=x.size(0))
        accuracy = (logits.argmax(1) == target).sum() / (target != self.pad_token).sum()
        self.log(f"accuracy{self.model.trafo_size}", accuracy, batch_size=x.size(0))
        return {f"loss{self.model.trafo_size}": loss, f"accuracy{self.model.trafo_size}": accuracy}

    def evaluate(self, dataset):
        loss_sum = 0.0
        accuracy_sum = 0.0
        for x in dataset:
            x = x["text"]
            x = torch.as_tensor([self.tokenizer.encode(x).ids], dtype=torch.int64)
            x = x.to(self.model.pos_embeddings.device)
            target = x[:, 1: self.model.context_length + 1]
            x = x[:, :min(self.model.context_length, x.size(1) - 1)]
            logits = self.model(x)
            logits = logits.transpose(-2, -1)
            loss = self.loss(logits, target)
            loss_sum += loss.cpu().item()
            accuracy = (logits.argmax(1) == target).sum() / target.size(1)
            accuracy_sum += accuracy.cpu().item()
        loss = loss_sum / len(dataset)
        accuracy = accuracy_sum / len(dataset)
        return loss, accuracy
