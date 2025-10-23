import matplotlib.pyplot as plt
import numpy as np


def calculate_pixel_perimeter(mask):
    """
    Правильный расчет периметра для пиксельной маски (4-связность)
    """
    h, w = mask.shape
    perimeter = 0

    for i in range(h):
        for j in range(w):
            if mask[i, j] > 0:  # Если пиксель принадлежит маске
                # Проверяем соседей (4-связность)
                neighbors = 0
                if i > 0 and mask[i - 1, j] > 0:
                    neighbors += 1
                if i < h - 1 and mask[i + 1, j] > 0:
                    neighbors += 1
                if j > 0 and mask[i, j - 1] > 0:
                    neighbors += 1
                if j < w - 1 and mask[i, j + 1] > 0:
                    neighbors += 1

                # Граничный пиксель имеет < 4 соседей
                perimeter += 4 - neighbors

    return perimeter


def calculate_compactness_old(mask):
    """Старый метод (неправильный)"""
    area = np.sum(mask)
    grad_x = np.abs(mask[:, 1:] - mask[:, :-1])
    grad_y = np.abs(mask[1:, :] - mask[:-1, :])
    perimeter = np.sum(grad_x) + np.sum(grad_y)
    return 4 * np.pi * area / (perimeter**2 + 1e-8)


def calculate_compactness_new(mask):
    """Новый метод (правильный)"""
    area = np.sum(mask)
    perimeter = calculate_pixel_perimeter(mask)
    return 4 * np.pi * area / (perimeter**2 + 1e-8)


# Создаем тестовые маски
def create_test_masks():
    masks = {}

    # 1. Круг
    y, x = np.ogrid[-25:25, -25:25]
    circle = np.zeros((50, 50))
    circle[x**2 + y**2 <= 15**2] = 1
    masks["Круг"] = circle

    # 2. Квадрат
    square = np.zeros((50, 50))
    square[10:40, 10:40] = 1
    masks["Квадрат"] = square

    # 3. Прямоугольник
    rectangle = np.zeros((50, 50))
    rectangle[15:35, 5:45] = 1
    masks["Прямоугольник"] = rectangle

    # 4. Полоски
    stripes = np.zeros((50, 50))
    for i in range(5):
        stripes[i * 10 : (i + 1) * 10, 10:40] = 1
    masks["Полоски"] = stripes

    # 5. Точки
    dots = np.zeros((50, 50))
    dots[10:40:15, 10:40:15] = 1
    masks["Точки"] = dots

    # 6. Кольцо (некомпактная форма)
    ring = np.zeros((50, 50))
    ring_mask = (x**2 + y**2 <= 20**2) & (x**2 + y**2 >= 10**2)
    ring[ring_mask] = 1
    masks["Кольцо"] = ring

    return masks


# Создаем маски
masks = create_test_masks()

# Визуализируем с сравнением методов
fig, axes = plt.subplots(3, 6, figsize=(18, 9))

for idx, (name, mask) in enumerate(masks.items()):
    # Расчет обоими методами
    compactness_old = calculate_compactness_old(mask)
    compactness_new = calculate_compactness_new(mask)

    area = np.sum(mask)
    perimeter_old = np.sum(np.abs(mask[:, 1:] - mask[:, :-1])) + np.sum(np.abs(mask[1:, :] - mask[:-1, :]))
    perimeter_new = calculate_pixel_perimeter(mask)

    # Основная маска
    axes[0, idx].imshow(mask, cmap="viridis")
    axes[0, idx].set_title(f"{name}\nПлощадь: {area:.0f}", fontsize=10)
    axes[0, idx].axis("off")

    # Старый метод периметра
    perimeter_mask_old = np.zeros_like(mask)
    perimeter_mask_old[:-1, :] += np.abs(mask[1:, :] - mask[:-1, :])
    perimeter_mask_old[:, :-1] += np.abs(mask[:, 1:] - mask[:, :-1])

    axes[1, idx].imshow(perimeter_mask_old, cmap="hot")
    axes[1, idx].set_title(
        f"Старый метод\nПериметр: {perimeter_old:.0f}\nКомпактность: {compactness_old:.3f}", fontsize=9
    )
    axes[1, idx].axis("off")

    # Новый метод периметра
    perimeter_mask_new = np.zeros_like(mask)
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j] > 0:
                neighbors = 0
                if i > 0 and mask[i - 1, j] > 0:
                    neighbors += 1
                if i < h - 1 and mask[i + 1, j] > 0:
                    neighbors += 1
                if j > 0 and mask[i, j - 1] > 0:
                    neighbors += 1
                if j < w - 1 and mask[i, j + 1] > 0:
                    neighbors += 1
                if neighbors < 4:
                    perimeter_mask_new[i, j] = 4 - neighbors

    axes[2, idx].imshow(perimeter_mask_new, cmap="hot")
    axes[2, idx].set_title(
        f"Новый метод\nПериметр: {perimeter_new:.0f}\nКомпактность: {compactness_new:.3f}", fontsize=9
    )
    axes[2, idx].axis("off")

plt.tight_layout()
plt.show()

# Вывод теоретических значений
print("=" * 60)
print("ТЕОРЕТИЧЕСКИЕ ЗНАЧЕНИЯ КОМПАКТНОСТИ:")
print("=" * 60)
print("Идеальный круг: 1.000")
print("Квадрат: π/4 ≈ 0.785")
print("Прямоугольник 2:1: ≈ 0.650")
print("Очень вытянутый: → 0.000")
print("=" * 60)

# Сравнительная таблица
print("\nСРАВНЕНИЕ МЕТОДОВ РАСЧЕТА:")
print("Форма".ljust(15) + "Старый".ljust(12) + "Новый".ljust(12) + "Разница")
print("-" * 50)
for name in masks.keys():
    mask = masks[name]
    old = calculate_compactness_old(mask)
    new = calculate_compactness_new(mask)
    diff = new - old
    print(f"{name.ljust(15)}{old:.3f}".ljust(12) + f"{new:.3f}".ljust(12) + f"{diff:+.3f}")
