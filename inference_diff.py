import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from model import Model
from dataloader import get_dataloader


class MultiModelAttentionVisualizer:
    def __init__(self, model_paths, data_dir, num_classes, model_names=None, batch_size=1, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(model_paths))]
        self.model_names = model_names

        # Загрузка моделей
        self.models = []
        for model_path in model_paths:
            model = Model(num_classes=num_classes, emb_dim=128)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            self.models.append(model)

        # Даталоадер
        train_loader, val_loader = get_dataloader(data_dir, batch_size=batch_size)
        self.val_loader = torch.utils.data.DataLoader(
            val_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.iterator = iter(self.val_loader)
        self.class_names = train_loader.dataset.classes

        # Настройка matplotlib - сетка 3x3
        self.fig, self.axes = plt.subplots(3, 3, figsize=(15, 12))
        plt.subplots_adjust(bottom=0.1, hspace=0.3, wspace=0.1)

        # Кнопка
        ax_button = plt.axes([0.4, 0.02, 0.2, 0.04])  # x, y, width, height
        self.button = Button(ax_button, "Next Image")
        self.button.on_clicked(self.next_image)

        # Для хранения изображений
        self.ims = [[None for _ in range(3)] for _ in range(3)]
        self.overlay_ims = [[None for _ in range(3)] for _ in range(3)]

        # Первый показ
        self.next_image(None)
        plt.show()

    def next_image(self, event):
        try:
            images, labels = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.val_loader)
            images, labels = next(self.iterator)

        images, labels = images.to(self.device), labels.to(self.device)

        # Денормализация изображения (общая для всех моделей)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = images[0].cpu() * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_np = img_denorm.permute(1, 2, 0).numpy()  # HWC

        true_label = labels[0].item()
        true_class_name = self.class_names[true_label]

        # Для каждой модели получаем attention maps
        attn_maps = []

        for model in self.models:
            with torch.no_grad():
                emb, attn_map = model(images, return_attn=True)
                attn_maps.append(attn_map[0:1])  # [1,1,H,W]

        # Обновление графиков для каждой модели
        for i, (model_name, attn_map) in enumerate(zip(self.model_names, attn_maps)):
            # Интерполяция mask
            mask_up = F.interpolate(
                attn_map, size=(img_denorm.shape[1], img_denorm.shape[2]), mode="bilinear", align_corners=False
            )
            mask_np = mask_up[0, 0].cpu().numpy()

            # Статистики внимания
            attn_mean = mask_np.mean()
            attn_max = mask_np.max()
            attn_min = mask_np.min()

            # Первый столбец: оригинальное изображение
            if self.ims[i][0] is None:
                self.ims[i][0] = self.axes[i, 0].imshow(img_np)
                if i == 0:
                    self.axes[i, 0].set_title(f"Original Image\nTrue: {true_class_name}", fontsize=10)
                else:
                    self.axes[i, 0].set_title(f"Original Image", fontsize=10)
            else:
                self.ims[i][0].set_data(img_np)
            self.axes[i, 0].axis("off")

            # Второй столбец: карта внимания
            if self.ims[i][1] is None:
                self.ims[i][1] = self.axes[i, 1].imshow(mask_np, cmap="jet", vmin=0, vmax=1)
                if i == 0:  # Colorbar только для первого ряда
                    cbar = self.fig.colorbar(self.ims[i][1], ax=self.axes[i, 1], shrink=0.8)
                    cbar.set_label("Attention Strength")
            else:
                self.ims[i][1].set_data(mask_np)

            self.axes[i, 1].set_title(f"{model_name}\n" f"mean={attn_mean:.3f}, max={attn_max:.3f}", fontsize=10)
            self.axes[i, 1].axis("off")

            # Третий столбец: наложение - ИСПРАВЛЕНИЕ ЗДЕСЬ
            if self.ims[i][2] is None:
                # Сначала отображаем оригинальное изображение
                self.ims[i][2] = self.axes[i, 2].imshow(img_np)
                # Затем накладываем маску внимания с прозрачностью
                self.overlay_ims[i][2] = self.axes[i, 2].imshow(mask_np, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            else:
                # Обновляем оба изображения
                self.ims[i][2].set_data(img_np)
                self.overlay_ims[i][2].set_data(mask_np)

            self.axes[i, 2].set_title(f"Overlay\n" f"min={attn_min:.3f}", fontsize=10)
            self.axes[i, 2].axis("off")

        self.fig.suptitle(f"Model Comparison - True class: {true_class_name}", fontsize=14)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    model_paths = [
        "results/model_epoch1.pth",
        "results/model_epoch5.pth",
        "results/model_epoch10.pth",
    ]
    # model_paths = [
    #     "results/prodfocusv1/model_epoch5.pth",
    #     "results/prodfocusv1/model_epoch10.pth",
    #     "results/prodfocusv1/model_epoch25.pth",
    # ]

    model_names = ["Epoch 1", "Epoch 4", "Epoch 10"]

    MultiModelAttentionVisualizer(
        model_paths=model_paths, model_names=model_names, data_dir="data", num_classes=124, batch_size=1
    )
