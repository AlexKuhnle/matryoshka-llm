import lightning
import sys
import torch
import torchvision
from torchvision.transforms import v2 as torchvision_transforms

from modules.cnn import CNN
from modules.vit import ViT
from vision_lightning import VisionLightning


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    num_epochs = int(sys.argv[3])

    transform = torchvision_transforms.Compose([
        torchvision_transforms.ToImage(),
        torchvision_transforms.ToDtype(torch.float32, scale=True),
    ])

    if dataset_name == "mnist":
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
    elif dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError

    input_size, *image_size = tuple(train_dataset[0][0].size())
    print(f"train: {len(train_dataset)}")
    print(f"test:  {len(test_dataset)}")
    print(f"shape: {image_size} x {input_size}")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    if model_name == "cnn":
        model_cls = CNN
    elif model_name == "vit":
        model_cls = ViT
    else:
        raise NotImplementedError
    model = VisionLightning(
        model=model_cls,
        model_kwargs=dict(input_size=input_size, output_size=10, image_size=image_size),
        learning_rate=1e-3,
    )

    logger = lightning.pytorch.loggers.TensorBoardLogger("lightning_logs_vision", name=f"{dataset_name}-{model_name}")
    trainer = lightning.Trainer(
        logger=logger,
        max_epochs=num_epochs,
        limit_val_batches=0.1,
        val_check_interval=0.1,
    )

    trainer.fit(model, train_dataloader, test_dataloader)
