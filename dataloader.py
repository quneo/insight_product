import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os


def get_dataloader(data_dir, batch_size=64, num_workers=2):
    """
    Создаёт train и val даталоадеры из предразделённых и предварительно аугментированных папок:
        data_dir/train/
        data_dir/val/

    Аугментации НЕ применяются — предполагается, что данные уже аугментированы.
    """

    # Одинаковые трансформации для train и val: только resize + нормализация
    common_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    assert os.path.isdir(train_dir), f"Train directory not found: {train_dir}"
    assert os.path.isdir(val_dir), f"Val directory not found: {val_dir}"

    # Оба датасета используют одинаковые трансформации — без аугментаций
    train_dataset = datasets.ImageFolder(root=train_dir, transform=common_tfms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=common_tfms)

    assert train_dataset.classes == val_dataset.classes, "Class names in train and val directories do not match!"

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Class to idx: {train_dataset.class_to_idx}")

    return train_loader, val_loader
