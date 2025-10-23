import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import uuid

# Поддерживаемые расширения
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def apply_simple_augmentation(img):
    """Применяет легкую аугментацию"""
    # Случайный поворот
    if random.random() > 0.5:
        angle = random.choice([-15, -10, -5, 5, 10, 15])
        img = img.rotate(angle)

    # Случайное отражение
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Легкое затемнение
    if random.random() > 0.7:
        factor = random.uniform(0.8, 0.95)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)

    # Легкий шум
    if random.random() > 0.8:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    return img


def augment_class_images(class_folder, target_count=400):
    """Аугментирует изображения в классе до target_count"""
    # Считаем текущие изображения
    images = [f for f in class_folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    current_count = len(images)

    if current_count >= target_count:
        print(f"✓ {class_folder.name}: {current_count} (достаточно)")
        return

    needed = target_count - current_count
    print(f"→ {class_folder.name}: {current_count} → {target_count}")

    # Загружаем оригинальные изображения
    original_imgs = []
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        original_imgs.append(img)

    # Создаем аугментированные
    for i in range(needed):
        original_img = random.choice(original_imgs)
        augmented_img = apply_simple_augmentation(original_img)

        new_name = f"aug_{uuid.uuid4().hex[:8]}.jpg"
        save_path = class_folder / new_name
        augmented_img.save(save_path, quality=95)

    print(f"✓ Добавлено {needed} изображений")


# Основная функция
def balance_dataset(data_folder, target_count=400):
    """Балансирует датасет"""
    data_path = Path(data_folder)

    for class_folder in data_path.iterdir():
        if class_folder.is_dir():
            augment_class_images(class_folder, target_count)


# Запуск
if __name__ == "__main__":
    balance_dataset(r"C:\Users\Nikitin Vadim\PythonCode\insight_product\freshdataset\weighter_dataset", 600)
