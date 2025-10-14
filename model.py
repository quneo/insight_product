import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchinfo import summary

class Model(nn.Module):
    def __init__(self, num_classes=None, emb_dim=128):
        super().__init__()
        backbone = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)

        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )

        self.attention_conv = nn.Conv2d(512, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.emb_fc = nn.Linear(512*7*7, emb_dim)

        if num_classes:
            self.classifier = nn.Linear(emb_dim, num_classes)
        else:
            self.classifier = None

    def forward(self, x, return_attn=False):
        features = self.features(x) # [B, 512, 7, 7]

        attn_map = self.sigmoid(self.attention_conv(features)) # [B, 1, 7, 7]

        features_attn = features * attn_map  # [B, 512, 7, 7]

        flatten_features = features_attn.view(features_attn.size(0), -1) # [B, 512 * 7 * 7]

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
    summary(model, input_size=(1, 3, 224, 224))