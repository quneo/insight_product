import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchinfo import summary


class Model(nn.Module):
    """
    ResNet-18 backbone + spatial attention (на 14x14).
    """

    def __init__(self, num_classes=None, emb_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Разбираем ResNet
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 56x56
        self.layer2 = backbone.layer2  # 28x28
        self.layer3 = backbone.layer3  # 14x14 ← внимание здесь
        self.layer4 = backbone.layer4  # 7x7

        # Spatial attention на 14×14 (выход layer3 → 256 каналов)
        self.spatial_attn = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1), nn.Sigmoid())

        # Эмбеддинг из последнего слоя (layer4 → 512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)

    def forward(self, x, return_attn=False):
        x = self.stem(x)  # [B, 64, 56, 56]
        x = self.layer1(x)  # [B, 64, 56, 56]
        x = self.layer2(x)  # [B, 128, 28, 28]
        feats_early = self.layer3(x)  # [B, 256, 14, 14]

        attn_map = self.spatial_attn(feats_early)  # [B,1,14,14]
        feats_weighted = feats_early * attn_map

        feats_final = self.layer4(feats_weighted)  # [B, 512, 7, 7]

        pooled = self.pool(feats_final).flatten(1)
        emb = F.normalize(self.bn(self.fc(pooled)), dim=1)

        if return_attn:
            return emb, attn_map
        else:
            return emb


if __name__ == "__main__":
    model = Model(num_classes=20)
    summary(model, input_size=(1, 3, 224, 224))
