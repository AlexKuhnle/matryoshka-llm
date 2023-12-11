import lightning
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2 as torchvision_transforms

from classifier_lightning import ClassifierLightning
from cnn.cnn import CNN


if __name__ == "__main__":

    transform = torchvision_transforms.Compose([
        torchvision_transforms.ToImage(),
        torchvision_transforms.ToDtype(torch.float32, scale=True),
    ])
    train_dataset = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    input_size, *image_size = tuple(train_dataset[0][0].size())
    print(f"train: {len(train_dataset)}")
    print(f"test:  {len(test_dataset)}")
    print(f"shape: {image_size} x {input_size}")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    model = ClassifierLightning(
        model=CNN,
        model_kwargs=dict(input_size=input_size, output_size=10, image_size=image_size),
        learning_rate=1e-3,
    )

    logger = lightning.pytorch.loggers.TensorBoardLogger("lightning_logs", name="cnn")
    trainer = lightning.Trainer(
        logger=logger,
        max_epochs=3,
        limit_val_batches=0.1,
        val_check_interval=0.1
    )

    trainer.fit(model, train_dataloader, test_dataloader)
