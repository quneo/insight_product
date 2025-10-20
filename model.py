import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchinfo import summary


class Model(nn.Module):
    """
    EfficientNet-B0 backbone + spatial attention (на 14x14).
    Используется встроенное SE (канальное внимание) + наше пространственное.
    """
    def __init__(self, num_classes=None, emb_dim=128):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        features = list(backbone.features.children())

        # EfficientNet-B0: features = [0..8] (9 блоков)
        # После features[5] → карта 14x14
        self.stage1 = nn.Sequential(*features[:6])  # до 14×14
        self.stage2 = nn.Sequential(*features[6:])  # после внимания — 7×7

        # Spatial attention на 14×14
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(112, 1, kernel_size=1),  # 112 каналов на этом уровне
            nn.Sigmoid()
        )

        # Эмбеддинг и классификация
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes) if num_classes else None

    def forward(self, x, return_attn=False):
        # До внимания
        feats_early = self.stage1(x)  # [B,112,14,14]

        # Пространственное внимание
        attn_map = self.spatial_attn(feats_early)  # [B,1,14,14]
        feats_weighted = feats_early * attn_map

        # Продолжение энкодера
        feats_final = self.stage2(feats_weighted)  # [B,1280,7,7]

        # Эмбеддинг
        pooled = self.pool(feats_final).flatten(1)
        emb = F.normalize(self.bn(self.fc(pooled)), dim=1)

        if self.classifier is not None:
            logits = self.classifier(emb)
            if return_attn:
                return emb, logits, attn_map
            else:
                return emb, logits
        else:
            if return_attn:
                return emb, attn_map
            else:
                return emb


if __name__ == "__main__":
    model = Model(num_classes=20)
    summary(model, input_size=(1, 3, 224, 224))
