import lightning
import torch
from typing import Callable


class ClassifierLightning(lightning.LightningModule):

    def __init__(
        self,
        model: Callable[..., torch.nn.Module],
        model_kwargs: dict,
        learning_rate: float,
    ):
        super().__init__()
        self.model = model(**model_kwargs)
        self.loss = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        logits = self.model(x)
        return logits.argmax(1)

    def training_step(self, batch, batch_idx):
        x, target = batch
        logits = self.model(x)
        loss = self.loss(logits, target)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, target = batch
        logits = self.model(x)
        loss = self.loss(logits, target)
        accuracy = (logits.argmax(1) == target).sum(0) / target.size(0)
        metrics = {"test_loss": loss, "accuracy": accuracy}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
