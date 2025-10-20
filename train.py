import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from losses import AttentionCrossEntropy
from model import Model  
from dataloader import get_dataloader

def train_model(
    model, 
    train_loader, 
    val_loader, 
    num_epochs=10, 
    lr=1e-4, 
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='results'
):
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = AttentionCrossEntropy()

    # Scheduler: уменьшает lr, если валидационный лосс не падает
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(1, num_epochs+1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # накопим статистику внимания и лоссов
        attn_mins, attn_means, attn_maxs = [], [], []
        loss_cls_vals, loss_center_vals, loss_compact_vals = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            emb, logits, attn_map = model(images, return_attn=True)
            loss, loss_dict = criterion(logits, labels, attn_map)
            loss.backward()
            optimizer.step()

            # статистика внимания
            attn_mins.append(attn_map.min().item())
            attn_means.append(attn_map.mean().item())
            attn_maxs.append(attn_map.max().item())

            # статистика лоссов
            loss_cls_vals.append(loss_dict["cls"])
            loss_center_vals.append(loss_dict["center"])
            loss_compact_vals.append(loss_dict["compact"])

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'cls': f"{loss_dict['cls']:.3f}",
                'center': f"{loss_dict['center']:.3f}",
                'compact': f"{loss_dict['compact']:.3f}",
                'attn_min': f"{attn_map.min().item():.3f}",
                'attn_mean': f"{attn_map.mean().item():.3f}",
                'attn_max': f"{attn_map.max().item():.3f}",
                'acc': f"{100*correct/total:.2f}%"
            })

        train_loss /= total
        train_acc = correct / total

        # Валидация
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                emb, logits, attn_map = model(images, return_attn=True)
                loss, _ = criterion(logits, labels, attn_map)

                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_acc = correct / total

        # Обновляем scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: Train loss={train_loss:.4f}, acc={train_acc:.4f} | Val loss={val_loss:.4f}, acc={val_acc:.4f} | LR={optimizer.param_groups[0]['lr']:.6f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Сохраняем модель каждые 3 эпохи
        if epoch % 3 == 0:
            save_path = os.path.join(save_dir, f'model_epoch{epoch}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")

    # Сохраняем финальную модель
    final_path = os.path.join(save_dir, 'model_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model: {final_path}")

    # Сохраняем графики
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_plot.png'))
    plt.show()

    return model, history

# Пример запуска
if __name__ == "__main__":
    train_loader, val_loader = get_dataloader("data/realdata", batch_size=32, val_split=0.2)
    model = Model(num_classes=len(train_loader.dataset.dataset.classes), emb_dim=128)
    trained_model, history = train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4)
