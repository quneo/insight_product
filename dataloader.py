import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import os
import matplotlib.pyplot as plt 


def get_dataloader(data_dir, batch_size=64, val_split=0.1, seed=41):
    """
    Создает train и validation даталоадеры для изображений
    
    Args:
        data_dir (str): путь к папке с данными
        batch_size (int): размер батча
        val_split (float): доля данных для валидации (0.0-1.0)
        seed (int): seed для воспроизводимости
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Определяем трансформации
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    
    # Загружаем весь датасет
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=train_tfms  # временно используем train трансформации
    )
    
    # Разделяем на train и validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    # Устанавливаем seed для воспроизводимости
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    # Применяем разные трансформации к train и val
    train_dataset.dataset.transform = train_tfms
    val_dataset.dataset.transform = val_tfms
    
    # Создаем даталоадеры
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Выводим информацию о данных
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {full_dataset.classes}")
    print(f"Class to idx: {full_dataset.class_to_idx}")
    
    return train_loader, val_loader

def show_batch(images, labels, class_names, denormalize=True):
    """
    Показывает батч изображений
    
    Args:
        images: тензор изображений (B, C, H, W)
        labels: тензор меток
        class_names: список имен классов
        denormalize: применять ли денормализацию
    """
    # Переводим в numpy и меняем порядок осей для matplotlib
    if denormalize:
        # Денормализуем изображения
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0, 1)
    
    # Преобразуем (B, C, H, W) -> (B, H, W, C) для matplotlib
    images = images.permute(0, 2, 3, 1).numpy()
    
    # Создаем grid для отображения
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    axes = axes.ravel()
    
    for i in range(min(len(images), 32)):
        axes[i].imshow(images[i])
        axes[i].set_title(class_names[labels[i]])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Пример использования
if __name__ == "__main__":
    train_loader, val_loader = get_dataloader(
        data_dir="data/realdata",
        batch_size=32,
        val_split=0.2
    )
    
    # Проверка первого батча
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        
        # Визуализация батча
        show_batch(images, labels, train_loader.dataset.classes)
        break