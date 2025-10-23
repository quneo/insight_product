import os
import random
import string
from pathlib import Path


def generate_random_name(length=16):
    """Генерирует случайное имя файла"""
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def rename_images_in_folder(root_dir, extensions=None):
    """
    Рекурсивно переименовывает все изображения в случайные имена

    Args:
        root_dir: корневая директория для поиска
        extensions: список расширений файлов (по умолчанию основные изображения)
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    renamed_count = 0
    error_count = 0

    print(f"🔍 Начинаем переименование в: {root_dir}")

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # Проверяем расширение файла
            file_ext = Path(file).suffix.lower()
            if file_ext in extensions:
                old_path = os.path.join(root, file)

                try:
                    # Генерируем новое имя с тем же расширением
                    new_name = generate_random_name() + file_ext
                    new_path = os.path.join(root, new_name)

                    # Переименовываем файл
                    os.rename(old_path, new_path)
                    renamed_count += 1

                    if renamed_count % 100 == 0:
                        print(f"Переименовано: {renamed_count} файлов...")

                except Exception as e:
                    print(f"Ошибка с {old_path}: {e}")
                    error_count += 1

    print(f"\nГотово!")
    print(f"Успешно переименовано: {renamed_count} файлов")
    print(f"Ошибок: {error_count}")


def preview_renaming(root_dir, sample_size=5):
    """Предпросмотр того, как будут переименованы файлы"""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    print("🔍 Предпросмотр переименования:")
    print("-" * 50)

    sample_count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_ext = Path(file).suffix.lower()
            if file_ext in extensions and sample_count < sample_size:
                new_name = generate_random_name() + file_ext
                print(f"'{file}' -> '{new_name}'")
                sample_count += 1
            if sample_count >= sample_size:
                break
        if sample_count >= sample_size:
            break

    print("-" * 50)


if __name__ == "__main__":
    TARGET_DIR = (
        r"C:\Users\Nikitin Vadim\PythonCode\insight_product\freshdataset\weighter_dataset"  # поменяй на свой путь
    )

    # Проверяем существование директории
    if not os.path.exists(TARGET_DIR):
        print(f"Директория {TARGET_DIR} не существует!")
        exit(1)

    # Предпросмотр
    preview_renaming(TARGET_DIR)

    # Подтверждение
    response = input("\n⚠️  Продолжить с переименованием? (y/n): ")
    if response.lower() != "y":
        print("Отменено пользователем")
        exit(0)

    # Запуск переименования
    rename_images_in_folder(TARGET_DIR)
