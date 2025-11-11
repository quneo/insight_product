import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchinfo import summary


class Model_v2(nn.Module):
    """
    ResNet-18 backbone + spatial attention (на 14x14).
    """

    def __init__(self, emb_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Разбираем ResNet
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 56x56
        self.layer2 = backbone.layer2  # 28x28
        self.layer3 = backbone.layer3  # 14x14

        # Spatial attention на 14×14
        self.spatial_attn = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())

        self.fc = nn.Linear(256, emb_dim)

    def forward(self, x, return_attn=False):
        x = self.stem(x)  # [B, 64, 56, 56]
        x = self.layer1(x)  # [B, 64, 56, 56]
        x = self.layer2(x)  # [B, 128, 28, 28]
        feats_early = self.layer3(x)  # [B, 256, 14, 14]

        attn_map = self.spatial_attn(feats_early)  # [B,1,14,14]
        feats_weighted = feats_early * attn_map  # [B, 256, 14, 14]

        feats_flat = torch.sum(feats_weighted, dim=(2, 3))  # [B, 256]

        emb = F.normalize(self.fc(feats_flat))

        if return_attn:
            return emb, attn_map
        else:
            return emb


if __name__ == "__main__":
    model = Model_v2()
    summary(model, input_size=(1, 3, 224, 224))
