"""Self-contained MiT-B0 + UPer decoder, compatible with the released weights."""
import torch
from torch import nn
from torch.nn import functional as F


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        noise = keep + torch.rand((len(x),) + (1,) * (x.ndim - 1), device=x.device, dtype=x.dtype)
        return x / keep * noise.floor()


class SpatialAttention(nn.Module):
    def __init__(self, channels, heads, reduction):
        super().__init__()
        self.head = heads
        self.sr_ratio = reduction
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, 2 * channels)
        self.proj = nn.Linear(channels, channels)
        if reduction > 1:
            self.sr = nn.Conv2d(channels, channels, reduction, reduction)
            self.norm = nn.LayerNorm(channels)

    def forward(self, tokens, height, width):
        batch, length, channels = tokens.shape
        q = self.q(tokens).reshape(batch, length, self.head, channels // self.head).permute(0, 2, 1, 3)
        reduced = tokens
        if self.sr_ratio > 1:
            spatial = tokens.permute(0, 2, 1).reshape(batch, channels, height, width)
            reduced = self.norm(self.sr(spatial).flatten(2).transpose(1, 2))
        k, v = self.kv(reduced).reshape(batch, -1, 2, self.head, channels // self.head).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(attended.transpose(1, 2).reshape(batch, length, channels))


class DepthwiseMix(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, tokens, height, width):
        batch, _, channels = tokens.shape
        spatial = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return self.dwconv(spatial).flatten(2).transpose(1, 2)


class MixMLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, 4 * channels)
        self.dwconv = DepthwiseMix(4 * channels)
        self.fc2 = nn.Linear(4 * channels, channels)

    def forward(self, x, height, width):
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), height, width)))


class TransformerBlock(nn.Module):
    def __init__(self, channels, heads, reduction, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = SpatialAttention(channels, heads, reduction)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = MixMLP(channels)

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        return x + self.drop_path(self.mlp(self.norm2(x), height, width))


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, image):
        spatial = self.proj(image)
        height, width = spatial.shape[-2:]
        return self.norm(spatial.flatten(2).transpose(1, 2)), height, width


class MiTB0(nn.Module):
    def __init__(self, first_stride=2):
        super().__init__()
        self.channels = [32, 64, 160, 256]
        rates = torch.linspace(0, 0.1, 8).tolist()
        for stage, (channels, heads, reduction) in enumerate(zip(self.channels, [1, 2, 5, 8], [8, 4, 2, 1]), 1):
            previous = 3 if stage == 1 else self.channels[stage - 2]
            setattr(self, f"patch_embed{stage}", PatchEmbedding(previous, channels, 7 if stage == 1 else 3, first_stride if stage == 1 else 2))
            setattr(self, f"block{stage}", nn.ModuleList([TransformerBlock(channels, heads, reduction, rates[2 * (stage - 1) + i]) for i in range(2)]))
            setattr(self, f"norm{stage}", nn.LayerNorm(channels))

    def forward(self, image):
        features = []
        for stage in range(1, 5):
            tokens, height, width = getattr(self, f"patch_embed{stage}")(image)
            for block in getattr(self, f"block{stage}"):
                tokens = block(tokens, height, width)
            image = getattr(self, f"norm{stage}")(tokens).reshape(len(tokens), height, width, -1).permute(0, 3, 1, 2)
            features.append(image)
        return features


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class PyramidPooling(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.stages = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(scale), ConvBlock(channels, width, 1)) for scale in (1, 2, 3, 6)])
        self.bottleneck = ConvBlock(channels + 4 * width, width, 3)

    def forward(self, feature):
        pooled = [F.interpolate(stage(feature), size=feature.shape[-2:], mode="bilinear", align_corners=True) for stage in self.stages]
        # Pooling-branch order is part of the released checkpoint format.
        return self.bottleneck(torch.cat([feature] + pooled[::-1], dim=1))


class UPerDecoder(nn.Module):
    def __init__(self, channels, width=128):
        super().__init__()
        self.ppm = PyramidPooling(channels[-1], width)
        self.fpn_in = nn.ModuleList([ConvBlock(c, width, 1) for c in channels[:-1]])
        self.fpn_out = nn.ModuleList([ConvBlock(width, width, 3) for _ in channels[:-1]])
        self.bottleneck = ConvBlock(4 * width, width, 3)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(width, 1, 1)

    def forward(self, features):
        value = self.ppm(features[-1])
        pyramid = [value]
        for i in range(2, -1, -1):
            feature = self.fpn_in[i](features[i])
            value = feature + F.interpolate(value, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            pyramid.append(self.fpn_out[i](value))
        pyramid.reverse()
        pyramid = [pyramid[0]] + [F.interpolate(p, size=pyramid[0].shape[-2:], mode="bilinear", align_corners=False) for p in pyramid[1:]]
        return self.conv_seg(self.dropout(self.bottleneck(torch.cat(pyramid, dim=1))))


class MiTSegmenter(nn.Module):
    def __init__(self, decoder_width=128, first_stride=2):
        super().__init__()
        self.encoder = MiTB0(first_stride)
        self.decoder = UPerDecoder(self.encoder.channels, decoder_width)

    def forward(self, image):
        return F.interpolate(self.decoder(self.encoder(image)), size=image.shape[-2:], mode="bilinear", align_corners=False)


class NativeRefinement(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.detail = nn.Sequential(
            nn.Conv2d(4, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.GELU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.GELU(), nn.Conv2d(24, 1, 1),
        )
        nn.init.zeros_(self.detail[-1].weight)
        nn.init.zeros_(self.detail[-1].bias)

    def forward(self, image):
        logits = self.base(image)
        return logits + self.detail(torch.cat([image, logits], dim=1))


def build_model(refinement=False, decoder_width=128, first_stride=2):
    base = MiTSegmenter(decoder_width, first_stride)
    return NativeRefinement(base) if refinement else base


def load_encoder(model, path):
    """Optional ImageNet MiT-B0 initialization; not needed for released full weights."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state.get("model", state))
    state = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith("head.")}
    base = model.base if isinstance(model, NativeRefinement) else model
    base.encoder.load_state_dict(state, strict=True)
