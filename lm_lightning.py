import lightning
import tokenizers
import torch
from typing import Callable


class LMLightning(lightning.LightningModule):

    def __init__(
        self,
        tokenizer: tokenizers.Tokenizer,
        model: Callable[..., torch.nn.Module],
        model_kwargs: dict,
        learning_rate: float,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.eos_token = self.tokenizer.token_to_id("</s>")
        self.model = model(**model_kwargs)
        self.loss = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        def assign_log_function_to_module(module, log_prefix):
            def fn_log(name, value):
                if self.model.training:
                    self.log(f"{log_prefix}-{name}", value, batch_size=1)
            assert not hasattr(module, "log")
            setattr(module, "log", fn_log)

        for module_name, module in self.model.named_modules():
            assign_log_function_to_module(module, f"gpt-{module_name}")

    def forward(self, x="", num_outputs=1, max_tokens=100):
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
            x = torch.nested.nested_tensor([
                torch.as_tensor(
                    self.tokenizer.encode(seq).ids[-self.model.context_length - 1: -1],
                    dtype=torch.int64,
                ).cuda() for seq in x
            ])

        eos = torch.zeros([x.size(0)], dtype=torch.bool).cuda()
        for _ in range(max_tokens):
            logits = self.model(x)[:, -1]
            next_token = logits.argmax(1)
            next_token = torch.where(eos, self.eos_token, next_token)
            x = torch.cat([x[-self.model.context_length + 1:], next_token.unsqueeze(1)], dim=1)
            eos = torch.logical_or(eos, next_token == self.eos_token)
            if eos.all().item():
                break

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
                x, batch_first=True, padding_value=self.eos_token,
            ).cuda()

        x, target = (
            x[:, :min(self.model.context_length, x.size(1) - 1)],
            x[:, 1: self.model.context_length + 1],
        )
        logits = self.model(x)
        logits = logits.transpose(-2, -1)
        loss = self.loss(logits, target)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if isinstance(batch["text"], str):
            batch["text"] = [batch["text"]]
        losses = list()
        accuracies = list()
        for x in batch["text"]:
            x = torch.as_tensor([self.tokenizer.encode(x).ids], dtype=torch.int64).cuda()
            x, target = (
                x[:, :min(self.model.context_length, x.size(1) - 1)],
                x[:, 1: self.model.context_length + 1],
            )
            logits = self.model(x)
            logits = logits.transpose(-2, -1)
            loss = self.loss(logits, target)
            losses.append(loss)
            accuracy = (logits.argmax(1) == target).sum() / target.numel()
            accuracies.append(accuracy)
        metrics = {
            "test_loss": torch.stack(losses).mean(),
            "accuracy": torch.stack(accuracies).mean(),
        }
        self.log_dict(metrics, batch_size=len(losses))
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
