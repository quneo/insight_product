import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from losses import ArcFaceLoss
from model import Model
from dataloader import get_dataloader


def fine_tune_model(
    model_path,
    train_loader,
    val_loader,
    num_epochs=10,
    lr=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="fine_tune_results",
):
    os.makedirs(save_dir, exist_ok=True)

    # Загрузка модели
    model = Model(num_classes=len(train_loader.dataset.classes), emb_dim=128)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Заморозка всех слоев кроме последних
    for name, param in model.named_parameters():
        # Замораживаем все слои кроме fc и bn
        if not any(layer in name for layer in ["fc", "layer4"]):
            param.requires_grad = False
        print(f"{name}: requires_grad = {param.requires_grad}")

    # Только ArcFace loss
    num_classes = len(train_loader.dataset.classes)
    criterion = ArcFaceLoss(num_classes=num_classes, emb_dim=128).to(device)

    if "arcface_state_dict" in checkpoint:
        criterion.load_state_dict(checkpoint["arcface_state_dict"])
        print("Веса ArcFace загружены")
    else:
        print("⚠️ Веса ArcFace не найдены в checkpoint")

    # Оптимизатор только для размороженных параметров
    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(criterion.parameters())
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("Начинаем дообучение с замороженными слоями...")
    print(f"Обучаемых параметров: {sum(p.numel() for p in trainable_params if p.requires_grad):,}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        criterion.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Fine-tune Epoch {epoch}/{num_epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            emb = model(images)  # Только эмбеддинг, без attention
            logits = criterion(emb, labels)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{100 * correct / total:.2f}%"})

        train_loss /= total
        train_acc = correct / total

        # Валидация
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Fine-tune Epoch {epoch}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                emb = model(images)
                logits = criterion(emb, labels)
                loss = torch.nn.functional.cross_entropy(logits, labels)

                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_acc = correct / total

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}: Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f}, acc={val_acc:.4f} | LR={optimizer.param_groups[0]['lr']:.6f}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Сохраняем чекпоинт каждую эпоху
        save_path = os.path.join(save_dir, f"fine_tuned_epoch{epoch}.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "arcface_state_dict": criterion.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            },
            save_path,
        )
        print(f"Saved checkpoint: {save_path} (val_acc: {val_acc:.4f})")

    # Финальная модель
    final_path = os.path.join(save_dir, "fine_tuned_final.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arcface_state_dict": criterion.state_dict(),
        },
        final_path,
    )
    print(f"Saved final fine-tuned model: {final_path}")

    # Графики
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Fine-tuning Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.title("Fine-tuning Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fine_tuning_plot.png"), dpi=150)
    plt.show()

    return model, history


if __name__ == "__main__":
    train_loader, val_loader = get_dataloader("data", batch_size=32)

    # Основной вариант
    trained_model, history = fine_tune_model(
        model_path="results/segment_tuned.pth",
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=15,
        lr=1e-4,
    )
