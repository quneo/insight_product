import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from losses import AttentionLoss
from model import Model
from dataloader import get_dataloader


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    lr=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="results",
):
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    num_classes = len(train_loader.dataset.classes)
    criterion = AttentionLoss(num_classes=num_classes, emb_dim=128).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        criterion.set_epoch(epoch)
        model.train()
        criterion.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            emb, attn_map = model(images, return_attn=True)
            loss, logits, loss_dict = criterion(emb, labels, attn_map)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            attn_min = attn_map.min().item()
            attn_mean = attn_map.mean().item()
            attn_max = attn_map.max().item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.3f}",
                    "cls": f"{loss_dict['cls']:.3f}",
                    "center": f"{loss_dict['center']:.3f}",
                    "compact": f"{loss_dict['compact']:.3f}",
                    "TV": f"{loss_dict['TV']:.3f}",
                    "attn_min": f"{attn_min:.3f}",
                    "attn_mean": f"{attn_mean:.3f}",
                    "attn_max": f"{attn_max:.3f}",
                    "acc": f"{100 * correct / total:.2f}%",
                }
            )

        train_loss /= total
        train_acc = correct / total

        model.eval()
        criterion.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                emb, attn_map = model(images, return_attn=True)
                loss, logits, _ = criterion(emb, labels, attn_map)

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

        save_path = os.path.join(save_dir, f"model_epoch{epoch}.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "arcface_state_dict": criterion.arcface.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            save_path,
        )
        print(f"Saved checkpoint: {save_path}")

    final_path = os.path.join(save_dir, "model_final.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arcface_state_dict": criterion.arcface.state_dict(),
        },
        final_path,
    )
    print(f"Saved final model: {final_path}")

    # Графики
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_plot.png"))
    plt.show()

    return model, history


# Пример запуска
if __name__ == "__main__":
    train_loader, val_loader = get_dataloader("data", batch_size=32)
    model = Model(num_classes=len(train_loader.dataset.classes), emb_dim=128)
    trained_model, history = train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4)
