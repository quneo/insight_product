# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, emb_dim, s=30.0, m=0.5, easy_margin=False):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(torch.pi - m))
        self.mm = torch.sin(torch.tensor(torch.pi - m)) * m

    def forward(self, emb, labels):
        # emb: [B, D], labels: [B]
        cosine = F.linear(emb, F.normalize(self.weight))  # [B, C]
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits


class AttentionLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        emb_dim=128,
        s=33.0,
        m=0.33,
        w_classif=1.0,
        w_center=0.6,
        w_compact=1.5,
        w_tv=1.0,
        w_sparse=2.5,
        w_max=1.5,
    ):
        super().__init__()
        self.arcface = ArcFaceLoss(num_classes, emb_dim, s=s, m=m)
        self.w_classif = w_classif
        self.w_center = w_center
        self.w_compact = w_compact
        self.w_tv = w_tv
        self.w_sparse = w_sparse
        self.w_max = w_max
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

        if epoch == 7:
            self.w_sparse = 0.1
            self.w_compact = 0.2
            self.w_tv = 0.1
            self.w_max = 0.4
            self.w_classif = 1.3

    def forward(self, emb, labels, attn_map):
        # emb: [B, D], labels: [B], attn_map: [B,1,H,W]
        logits = self.arcface(emb, labels)
        B, _, H, W = attn_map.shape
        eps = 1e-8

        # --- 1. ArcFace классификация ---
        L_cls = F.cross_entropy(logits, labels)

        # --- 2. Центральный лосс ---
        # Создаем маску центра (гауссово распределение)
        if self.current_epoch <= 3 and self.w_center > 0:
            y = torch.linspace(-1, 1, H, device=attn_map.device).view(H, 1)
            x = torch.linspace(-1, 1, W, device=attn_map.device).view(1, W)
            dist = torch.sqrt(x**2 + y**2)  # евклидово расстояние от центра
            center_mask = torch.exp(-dist * 3)  # гауссово ядро, 3 - sharpness

            # Поощряем внимание в центре, штрафуем на краях
            L_center = (
                1.0 - F.cosine_similarity(attn_map.view(B, -1), center_mask.view(1, -1).expand(B, -1), dim=1).mean()
            )
        else:
            L_center = torch.tensor(0.0, device=attn_map.device)

        # --- 3. Компактность ---
        # Штрафуем за распыленное внимание
        if self.current_epoch <= 5:
            L_compact = -torch.mean(attn_map * torch.log(attn_map + eps))  # энтропия - чем концентрированнее, тем лучше
        else:
            L_compact = torch.tensor(0.0, device=attn_map.device)

        # --- 4. Гладкость (Total Variation) ---
        diff_h = attn_map[:, :, 1:, :] - attn_map[:, :, :-1, :]
        diff_w = attn_map[:, :, :, 1:] - attn_map[:, :, :, :-1]
        L_tv = torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_w))

        # --- 5. Sparsity: штраф за большое среднее ---
        mu = attn_map.mean()
        if self.current_epoch <= 7:
            L_sparse = torch.relu(mu - 0.35)
        else:
            L_sparse = torch.tensor(0.0, device=attn_map.device)

        # --- 6. Штраф за низкие максимумы ---
        attn_max_vals = attn_map.view(B, -1).max(dim=1).values  # [B]
        L_max = torch.mean(torch.relu(0.9 - attn_max_vals) ** 2)

        # --- Суммарный лосс ---
        L_total = (
            self.w_classif * L_cls
            + self.w_center * L_center
            + self.w_compact * L_compact
            + self.w_tv * L_tv
            + self.w_sparse * L_sparse
            + self.w_max * L_max
        )

        return (
            L_total,
            logits,
            {
                "total": L_total.item(),
                "cls": L_cls.item(),
                "center": L_center.item(),
                "compact": L_compact.item(),
                "TV": L_tv.item(),
                "sparse": L_sparse.item(),
            },
        )
