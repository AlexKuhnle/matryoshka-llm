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

    def forward(self, x="", num_outputs=1, max_tokens=100):
        assert x == "" or num_outputs == 1
        if num_outputs > 1:
            x = torch.as_tensor(
                [self.tokenizer.encode("").ids[-self.model.context_size: -1]],
                dtype=torch.int64,
            ).repeat(num_outputs, 1).cuda()
        else:
            x = torch.as_tensor(
                [self.tokenizer.encode(x).ids[-self.model.context_size: -1]],
                dtype=torch.int64,
            ).cuda()

        eos = torch.zeros([x.size(0)], dtype=torch.bool).cuda()
        for _ in range(max_tokens):
            logits = self.model(x)[:, -1]
            next_token = logits.argmax(1)
            next_token = torch.where(eos, self.eos_token, next_token)
            x = torch.cat([x, next_token.unsqueeze(1)], dim=1)
            eos = torch.logical_or(eos, next_token == self.eos_token)
            if eos.all().item():
                break

        if num_outputs == 1:
            return self.tokenizer.decode(x[0].tolist())
        else:
            return [self.tokenizer.decode(seq.tolist()) for seq in x]

    def training_step(self, batch, batch_idx):
        x = batch["text"]
        x = torch.as_tensor(
            [self.tokenizer.encode(x).ids[:self.model.context_size + 1]],
            dtype=torch.int64,
        ).cuda()
        x, target = x[:, :-1], x[:, 1:]
        logits = self.model(x)
        logits = logits.transpose(-2, -1)
        loss = self.loss(logits, target)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch["text"]
        x = torch.as_tensor(
            [self.tokenizer.encode(x).ids[:self.model.context_size + 1]],
            dtype=torch.int64,
        ).cuda()
        x, target = x[:, :-1], x[:, 1:]
        logits = self.model(x)
        logits = logits.transpose(-2, -1)
        loss = self.loss(logits, target)
        accuracy = (logits.argmax(1) == target).sum() / target.numel()
        metrics = {"test_loss": loss, "accuracy": accuracy}
        self.log_dict(metrics, batch_size=x.size(0))
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
