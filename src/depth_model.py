"""
Pure PyTorch implementation of DINOv2 ViT-Small + DPT Depth Anything.
Loaded directly from local safetensors checkpoints without network dependencies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from pathlib import Path
from typing import Union

class ResidualConvUnit(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x

class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            if output.shape[-2:] != res.shape[-2:]:
                output = F.interpolate(output, size=res.shape[-2:], mode="bilinear", align_corners=True)
            output = output + res
        output = self.resConfUnit2(output)
        output = self.out_conv(output)
        return output

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

class Attention(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ls1 = nn.Module()
        self.ls1.gamma = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.ls2 = nn.Module()
        self.ls2.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1.gamma * self.attn(self.norm1(x))
        x = x + self.ls2.gamma * self.mlp(self.norm2(x))
        return x

class DINOv2ViTSmall(nn.Module):
    def __init__(self, img_size: int = 518, patch_size: int = 14, embed_dim: int = 384, depth: int = 12, num_heads: int = 6):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size)**2 + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward_features(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Positional embedding interpolation if resolution != 518
        if x.shape[1] != self.pos_embed.shape[1]:
            cls_pos = self.pos_embed[:, :1]
            patch_pos = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            h_old = w_old = int((patch_pos.shape[1]) ** 0.5)
            h_new = H // self.patch_size
            w_new = W // self.patch_size
            patch_pos = patch_pos.reshape(1, h_old, w_old, dim).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(h_new, w_new), mode="bicubic", align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
            pos_embed = torch.cat([cls_pos, patch_pos], dim=1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed
        
        intermediate_features = []
        out_indices = [2, 5, 8, 11]
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in out_indices:
                intermediate_features.append(x)
        return intermediate_features

class DPTHead(nn.Module):
    def __init__(self, in_channels=[48, 96, 192, 384], features=64, out_channels=[32, 32]):
        super().__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(384, in_channels[0], kernel_size=1, bias=True),
            nn.Conv2d(384, in_channels[1], kernel_size=1, bias=True),
            nn.Conv2d(384, in_channels[2], kernel_size=1, bias=True),
            nn.Conv2d(384, in_channels[3], kernel_size=1, bias=True),
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels[0], in_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(in_channels[1], in_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(in_channels[3], in_channels[3], kernel_size=3, stride=2, padding=1),
        ])
        
        self.scratch = nn.Module()
        self.scratch.layer1_rn = nn.Conv2d(in_channels[0], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.scratch.layer2_rn = nn.Conv2d(in_channels[1], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.scratch.layer3_rn = nn.Conv2d(in_channels[2], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.scratch.layer4_rn = nn.Conv2d(in_channels[3], features, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        
        self.scratch.output_conv1 = nn.Conv2d(features, out_channels[0], kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channels[1], 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )

    def forward(self, out_features, patch_h: int, patch_w: int) -> torch.Tensor:
        out = []
        for i, x in enumerate(out_features):
            x = x[:, 1:, :].permute(0, 2, 1).reshape(x.shape[0], -1, patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        return out

class DepthAnything(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = DINOv2ViTSmall()
        self.depth_head = DPTHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.forward_features(x)
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, (H, W), mode="bilinear", align_corners=True)
        return depth

def load_depth_model(weights_path: Union[str, Path], device: torch.device) -> DepthAnything:
    model = DepthAnything()
    weights_path = Path(weights_path)
    if weights_path.suffix == ".safetensors":
        state_dict = load_file(str(weights_path), device="cpu")
    else:
        state_dict = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model
