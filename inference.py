import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from model import Model
from dataloader import get_dataloader


class AttentionVisualizer:
    def __init__(self, model_path, data_dir, num_classes, batch_size=1, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Модель
        self.model = Model(num_classes=num_classes, emb_dim=128)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        # Даталоадер
        train_loader, val_loader = get_dataloader(data_dir, batch_size=batch_size)
        self.val_loader = torch.utils.data.DataLoader(
            val_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.iterator = iter(self.val_loader)
        self.class_names = train_loader.dataset.classes

        # Настройка matplotlib
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(bottom=0.2)  # место для кнопки

        # Кнопка
        ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])  # x, y, width, height
        self.button = Button(ax_button, "Next Image")
        self.button.on_clicked(self.next_image)

        # Для colorbar
        self.im2 = None  # для карты внимания

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
        with torch.no_grad():
            emb, attn_map = self.model(images, return_attn=True)

        img = images[0].cpu()
        mask = attn_map[0:1]  # [1,1,H,W]

        # Интерполяция mask
        mask_up = F.interpolate(mask, size=(img.shape[1], img.shape[2]), mode="bilinear", align_corners=False)
        mask_np = mask_up[0, 0].cpu().numpy()

        # Денормализация изображения
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_np = img_denorm.permute(1, 2, 0).numpy()  # HWC

        # Обновление графиков
        self.axes[0].imshow(img_np)
        self.axes[0].set_title("Original Image")
        self.axes[0].axis("off")

        # Attention map с colorbar
        if self.im2 is None:
            self.im2 = self.axes[1].imshow(mask_np, cmap="jet", vmin=0, vmax=1)
            cbar = self.fig.colorbar(self.im2, ax=self.axes[1])
        else:
            self.im2.set_data(mask_np)

        self.axes[1].set_title("Attention Map")
        self.axes[1].axis("off")

        self.axes[2].imshow(img_np)
        self.axes[2].imshow(mask_np, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        self.axes[2].set_title("Overlay")
        self.axes[2].axis("off")

        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    AttentionVisualizer(
        model_path=r"C:\Users\Nikitin Vadim\PythonCode\insight_product\results\square\square_model_epoch100.pth",
        data_dir="data",
        num_classes=124,
        batch_size=1,
    )
