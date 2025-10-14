# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCrossEntropy(nn.Module):
    def __init__(self, w_classif=1.0, w_center=1.3, w_compact=1.5):
        super().__init__()
        self.w_classif = w_classif
        self.w_center = w_center
        self.w_compact = w_compact

    def forward(self, logits, labels, attn_logits):
        mask = torch.sigmoid(attn_logits)  # [B,1,H,W]
        B, _, H, W = mask.shape
        eps = 1e-8

        # --- 1. Основная классификация ---
        L_cls = F.cross_entropy(logits, labels)

        # --- 2. Центральный bias ---
        # Создаем маску центра (гауссово распределение)
        y = torch.linspace(-1, 1, H, device=mask.device).view(H, 1)
        x = torch.linspace(-1, 1, W, device=mask.device).view(1, W)
        dist = torch.sqrt(x**2 + y**2)  # евклидово расстояние от центра
        center_mask = torch.exp(-dist * 3)  # гауссово ядро, 3 - sharpness
        
        # Поощряем внимание в центре, штрафуем на краях
        L_center = 1.0 - F.cosine_similarity(
            mask.view(B, -1), 
            center_mask.view(1, -1).expand(B, -1), 
            dim=1
        ).mean()

        # --- 3. Компактность ---
        # Штрафуем за распыленное внимание
        L_compact = -torch.mean(mask * torch.log(mask + eps))  # энтропия - чем концентрированнее, тем лучше

        # --- Суммарный лосс ---
        L_total = (self.w_classif * L_cls + 
                  self.w_center * L_center + 
                  self.w_compact * L_compact)

        return L_total, {
            "total": L_total.item(),
            "cls": L_cls.item(),
            "center": L_center.item(),
            "compact": L_compact.item()
        }