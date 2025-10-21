import os
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms

# Поддерживаемые расширения
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Усиленная аугментация
augment_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),  # чуть больше, чтобы потом обрезать
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
        transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Обратно в PIL для сохранения
        # transforms.Lambda(lambda x: transforms.ToPILImage()(x)),
    ]
)


def augment_and_save_images(class_folder: Path, target_count: int = 400):
    # Найти все изображения
    image_files = [f for f in class_folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file()]
    current_count = len(image_files)

    if current_count >= target_count:
        print(f"⏭️  Пропущено: {class_folder.name} ({current_count} ≥ {target_count})")
        return

    needed = target_count - current_count
    print(f"🔄 Аугментация: {class_folder.name} — нужно {needed} изображений (сейчас {current_count})")

    # Убедимся, что выходной формат — RGB (PIL не любит RGBA при сохранении в JPEG)
    def ensure_rgb(img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    # Загружаем все исходные изображения один раз
    original_images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
            original_images.append(img)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {img_path}: {e}")

    if not original_images:
        print(f"❌ Нет валидных изображений в {class_folder}")
        return

    # Генерируем недостающие изображения
    for i in tqdm(range(needed), desc=f"Аугментация {class_folder.name}"):
        # Случайно выбираем исходное изображение
        src_img = random.choice(original_images)
        augmented_img = augment_transform(src_img)
        augmented_img = ensure_rgb(augmented_img)

        # Генерируем уникальное имя
        new_name = f"aug_{uuid.uuid4().hex}.jpg"
        save_path = class_folder / new_name

        # Сохраняем (в JPEG для компактности)
        augmented_img.save(save_path, quality=95)

    print(f"✅ Добавлено {needed} аугментированных изображений в {class_folder.name}")


# === ОСНОВНАЯ ЧАСТЬ ===
import uuid


def balance_dataset(root_folder: str, min_threshold: int = 100, target_count: int = 400):
    root = Path(root_folder)
    if not root.is_dir():
        raise ValueError(f"Папка не найдена: {root_folder}")

    # Получаем все подпапки (классы)
    class_folders = [d for d in root.iterdir() if d.is_dir()]

    print(f"📁 Найдено {len(class_folders)} классов. Начинаем аугментацию...")

    for folder in sorted(class_folders):
        image_count = len([f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file()])
        if image_count < min_threshold:
            augment_and_save_images(folder, target_count=target_count)
        else:
            print(f"⏭️  Пропущено: {folder.name} ({image_count} ≥ {min_threshold})")


# === ЗАПУСК ===
if __name__ == "__main__":
    ROOT_DATASET = r"C:\Users\Nikitin Vadim\Desktop\weighter_dataset_aug"
    balance_dataset(ROOT_DATASET, min_threshold=390, target_count=400)
