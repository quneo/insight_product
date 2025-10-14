# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCrossEntropy(nn.Module):
    """
    CrossEntropy + маленький энтропийный стабилизатор на sigmoid(attn_logits)
    + регуляризаторы для карты внимания:
        - центральный bias (объект чаще в центре)
        - компактность/спарсность внимания
    attn_logits: raw logits [B,1,H,W] (sa_logits)
    """
    def __init__(self, w_classif=0.6, w_entropy=0.6, alpha_center=1.5, beta_compact=0.8):
        super().__init__()
        self.w_classif = w_classif
        self.w_entropy = w_entropy
        self.alpha_center = alpha_center
        self.beta_compact = beta_compact

    def forward(self, logits, labels, attn_logits):
        # --- 1. Классификация ---
        L_cls = F.cross_entropy(logits, labels)

        # --- 2. Энтропийный стабилизатор карты внимания ---
        mask = attn_logits  # [B,1,H,W], после сигмоиды
        eps = 1e-8
        entropy = - (mask * torch.log(mask + eps) + (1 - mask) * torch.log(1 - mask + eps))
        L_entropy = entropy.mean()

        # --- 3. Центральный bias ---
        B, _, H, W = mask.shape
        y = torch.linspace(-1, 1, H, device=mask.device).view(H,1)
        x = torch.linspace(-1, 1, W, device=mask.device).view(1,W)
        dist = x**2 + y**2
        center_mask = 1 - dist / dist.max()  # максимум в центре
        B = mask.shape[0]
        center_mask_batch = center_mask.unsqueeze(0).expand(B, -1, -1)  # [B,H,W]
        L_center = F.mse_loss(mask.squeeze(1), center_mask_batch)

        # --- 4. Компактность / спарсность ---
        L_compact = -torch.mean(mask * torch.log(mask + eps))

        # --- 5. Суммарный лосс ---
        L_total = self.w_classif * L_cls + self.w_entropy * L_entropy + self.alpha_center * L_center + self.beta_compact * L_compact

        return L_total, {
            "total": L_total.item(),
            "cls": L_cls.item(),
            "entropy": L_entropy.item(),
            "center": L_center.item(),
            "compact": L_compact.item()
        }
