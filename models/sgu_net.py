import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# ========================= CBAM Module =========================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        mid = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(in_planes, mid, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid, in_planes, 1, bias=False)
        )

    def forward(self, x):
        return torch.sigmoid(self.shared(self.avg_pool(x)) + self.shared(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)


# ========================= SVM-Guided Adaptive Gate =========================
class SVMGuidedGate(nn.Module):
    def __init__(self, in_channels):
        super(SVMGuidedGate, self).__init__()
        self.svm_conv = nn.Conv2d(1, in_channels, kernel_size=1)
        self.cbam = CBAM(in_channels)

    def forward(self, feature, svm_map):
        # resize SVM map
        svm_resized = F.interpolate(svm_map, size=feature.shape[2:], mode='bilinear', align_corners=False)

        # SVM attention map
        svm_att = torch.sigmoid(self.svm_conv(svm_resized))  # [B, C, H, W]

        # Weighting logic based on range
        # Soft weighting: values <0.3 → 0, 0.3–0.7 → linear, >0.7 → 1
        weight_map = torch.clamp((svm_att - 0.3) / 0.4, 0, 1)  # normalizes to [0,1]

        # Feature modulation
        gated = feature * (1 + weight_map)

        # CBAM to enhance relevant attention
        gated = self.cbam(gated)

        return gated


# ========================= Final Model =========================
class SVMGuidedAttentionUNet(nn.Module):
    def __init__(self, encoder_name='resnet50', encoder_weights='imagenet', classes=2, activation='sigmoid'):
        super(SVMGuidedAttentionUNet, self).__init__()

        # Base U-Net
        self.base_unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation
        )

        self.encoder = self.base_unet.encoder
        self.decoder = self.base_unet.decoder
        self.segmentation_head = self.base_unet.segmentation_head

        # Only modulate lower-level feature (encoder[0])
        self.guided_gate = SVMGuidedGate(self.encoder.out_channels[0])

    def forward(self, x, svm_map=None):
        features = self.encoder(x)

        if svm_map is not None:
            features[0] = self.guided_gate(features[0], svm_map)
        else:
            features[0] = CBAM(self.encoder.out_channels[0])(features[0])

        decoder_output = self.decoder(*features)
        return self.segmentation_head(decoder_output)



