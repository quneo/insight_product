import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchinfo import summary

class Model(nn.Module):
    def __init__(self, num_classes=None, emb_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Разделяем backbone на части, убираем последний ReLU из layer3
        layer3_modules = list(backbone.layer3.children())
        last_block = layer3_modules[-1]
        # Уберем ReLU из последнего Bottleneck блока
        last_block.relu = nn.Identity()
        layer3_modules[-1] = last_block

        self.early_features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, nn.Sequential(*layer3_modules)
        )
        
        self.layer4 = backbone.layer4  # отдельно layer4

        # Attention после layer3 (14x14)
        self.attention_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.post_attn_relu = nn.ReLU(inplace=True)  # ReLU после применения attention

        self.emb_fc = nn.Linear(512*7*7, emb_dim)  # финальный размер после layer4

        if num_classes:
            self.classifier = nn.Linear(emb_dim, num_classes)
        else:
            self.classifier = None

    def forward(self, x, return_attn=False):
        # Ранние features
        features_early = self.early_features(x)  # [B, 256, 14, 14]

        # Attention
        attn_map = self.sigmoid(self.attention_conv(features_early))  # [B,1,14,14]

        # Применяем attention
        features_attn = features_early * attn_map
        features_attn = self.post_attn_relu(features_attn)  # ReLU после взвешивания

        # Layer4
        features_final = self.layer4(features_attn)  # [B,512,7,7]

        # Классификация
        flatten_features = features_final.view(features_final.size(0), -1)
        emb = self.emb_fc(flatten_features)

        if self.classifier is not None:
            out = self.classifier(emb)
            if return_attn:
                return emb, out, attn_map
            else:
                return emb, out
        else:
            if return_attn:
                return emb, attn_map
            else:
                return emb

            

if __name__ == "__main__":
    model = Model(num_classes=20)
    summary(model)