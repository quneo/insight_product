import os
import random
import shutil
from pathlib import Path

# Поддерживаемые расширения
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def split_dataset(
    root_folder: str,
    val_size: int = 70,
    seed: int = 42,
    move_files: bool = True,  # True = переместить, False = копировать
):
    root = Path(root_folder)
    if not root.is_dir():
        raise ValueError(f"❌ Исходная папка не найдена: {root_folder}")

    train_dir = root.parent / f"{root.name}_train"
    val_dir = root.parent / f"{root.name}_val"

    # Создаём целевые папки
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    print(f"📁 Исходная папка: {root}")
    print(f"📤 Train: {train_dir}")
    print(f"📥 Val: {val_dir}")
    print(f"🎲 Seed: {seed}, Val size per class: {val_size}")
    print("-" * 50)

    # Фиксируем seed для воспроизводимости
    random.seed(seed)

    class_folders = [d for d in root.iterdir() if d.is_dir()]
    class_folders.sort()  # чтобы порядок был одинаковый

    for class_folder in class_folders:
        class_name = class_folder.name
        print(f"\n🔄 Обработка класса: {class_name}")

        # Находим все изображения
        image_files = [f for f in class_folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

        if not image_files:
            print(f"⚠️  Пропущено: нет изображений в {class_name}")
            continue

        # Сортируем для детерминированности (на случай, если glob возвращает в разном порядке)
        image_files.sort()

        total = len(image_files)
        print(f"   Всего изображений: {total}")

        if total <= val_size:
            print(f"   ⚠️ Мало данных: все {total} изображений идут в val")
            val_files = image_files
            train_files = []
        else:
            # Детерминированный случайный выбор
            val_files = random.sample(image_files, val_size)
            train_files = [f for f in image_files if f not in val_files]

        # Создаём папки классов в train и val
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)

        # Функция для копирования/перемещения
        op = shutil.move if move_files else shutil.copy

        # Перемещаем/копируем в val
        for f in val_files:
            op(str(f), str(val_dir / class_name / f.name))

        # Перемещаем/копируем в train
        for f in train_files:
            op(str(f), str(train_dir / class_name / f.name))

        print(f"   ✅ Val: {len(val_files)}, Train: {len(train_files)}")

    print("\n✨ Разделение завершено!")
    print(f"📁 Train: {train_dir}")
    print(f"📁 Val: {val_dir}")


# === ЗАПУСК ===
if __name__ == "__main__":
    # Укажи путь к папке с классами (например, "dataset_balanced")
    DATASET_ROOT = (
        r"C:\Users\Nikitin Vadim\PythonCode\insight_product\data\weighter_dataset"  # ← ЗАМЕНИ НА СВОЙ ПУТЬ!
    )

    split_dataset(
        root_folder=DATASET_ROOT,
        val_size=70,
        seed=42,  # ← фиксированный seed для воспроизводимости
        move_files=True,  # True = переместить файлы, False = оставить оригинал и скопировать
    )
