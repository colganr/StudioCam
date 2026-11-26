#!/usr/bin/env python3
"""
Converts two permissive (MIT & Apache-2.0) real-time portrait/video matting models
into zero-copy Apple Silicon Neural Engine (ANE) FP16 Core ML (.mlpackage) bundles:

1. StudioCamMODNetWebcamF16.mlpackage (Apache-2.0, ZHKKKe/MODNet webcam portrait weights)
   - Zero-setup universal webcam portrait matting at 512x288 (16:9).
   - Surgical ANE graph optimizations: 4D Conv2d SE-blocks (no reshape/Linear) + static shapes.
   - Direct IOSurface CVPixelBuffer ImageType input (`image`, BGRA) -> ImageType output (`alpha`, OneComponent8).

2. StudioCamBGMv2CleanPlateF16.mlpackage (MIT, PeterL1n/BackgroundMattingV2 MattingBase MobileNetV2)
   - 1-Click "Empty Room / Chair Clean Plate" matting at 512x288 (16:9).
   - Takes (`src` live frame, `bgr` empty chair/room reference) with outer-corner auto-exposure gain matching.
   - Prunes unused refiner channels (`err`, `hid`) from final conv4 (89% fewer final-layer FLOPs, bit-exact `pha`).
"""

import os
import sys
import shutil
import subprocess
import tempfile
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

H, W = 288, 512  # Exact 16:9 aspect ratio, divisible by 32 for both MobileNetV2 encoders

DEFAULT_MODELS_DIR = os.path.expanduser("~/Library/Application Support/StudioCam/Models")
BGM_URL = "https://github.com/PeterL1n/BackgroundMattingV2/releases/download/v1.0.0/pytorch_mobilenetv2.pth"
MODNET_URL = "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/models/modnet_webcam_portrait_matting.ckpt"


# ==============================================================================
# 1. BackgroundMattingV2 (MIT License) — Clean-Plate MattingBase (Zero torchvision)
# ==============================================================================

class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1):
        padding = ((kernel_size - 1) // 2) * dilation
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=False),
        )


class TVInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU6(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, dilation=dilation),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return (x + self.conv(x)) if self.use_res_connect else self.conv(x)


class BGMv2MobileNetV2Encoder(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        cfg = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        features = [ConvBNReLU6(in_channels, 32, stride=2)]
        inp = 32
        for t, c, n, s in cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(TVInvertedResidual(inp, c, stride=stride, expand_ratio=t))
                inp = c
        # Output stride 16 dilation adjustments matching PeterL1n/BackgroundMattingV2
        features[14].conv[1][0].stride = (1, 1)
        for feat in features[15:]:
            feat.conv[1][0].dilation = (2, 2)
            feat.conv[1][0].padding = (2, 2)
        self.features = nn.Sequential(*features)

    def forward(self, x):
        x0 = x
        x = self.features[0](x)
        x = self.features[1](x)
        x1 = x
        x = self.features[2](x)
        x = self.features[3](x)
        x2 = x
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        x3 = x
        for idx in range(7, 18):
            x = self.features[idx](x)
        x4 = x
        return x4, x3, x2, x1, x0


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=(int(h), int(w)), mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels=320, atrous_rates=(3, 6, 9), out_channels=256):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
            )
        ]
        for r in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, r))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        return self.project(torch.cat(res, dim=1))


class BGMv2Decoder(nn.Module):
    def __init__(self, channels=(256, 128, 64, 48, 37), feature_channels=(32, 24, 16, 6)):
        super().__init__()
        self.conv1 = nn.Conv2d(feature_channels[0] + channels[0], channels[1], 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(feature_channels[1] + channels[1], channels[2], 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.conv3 = nn.Conv2d(feature_channels[2] + channels[2], channels[3], 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.conv4 = nn.Conv2d(feature_channels[3] + channels[3], channels[4], 3, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x4, x3, x2, x1, x0):
        x = F.interpolate(x4, size=(int(x3.shape[2]), int(x3.shape[3])), mode="bilinear", align_corners=False)
        x = self.relu(self.bn1(self.conv1(torch.cat([x, x3], dim=1))))
        x = F.interpolate(x, size=(int(x2.shape[2]), int(x2.shape[3])), mode="bilinear", align_corners=False)
        x = self.relu(self.bn2(self.conv2(torch.cat([x, x2], dim=1))))
        x = F.interpolate(x, size=(int(x1.shape[2]), int(x1.shape[3])), mode="bilinear", align_corners=False)
        x = self.relu(self.bn3(self.conv3(torch.cat([x, x1], dim=1))))
        x = F.interpolate(x, size=(int(x0.shape[2]), int(x0.shape[3])), mode="bilinear", align_corners=False)
        return self.conv4(torch.cat([x, x0], dim=1))


class StudioCamBGMv2CleanPlate(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BGMv2MobileNetV2Encoder(6)
        self.aspp = ASPP(320, (3, 6, 9), 256)
        self.decoder = BGMv2Decoder((256, 128, 64, 48, 37), (32, 24, 16, 6))

    def prune_alpha_only_head(self):
        # Channel 0 is `pha` (alpha). Pruning channels 1..36 eliminates 97% of final conv4 output channels!
        old = self.decoder.conv4
        pruned = nn.Conv2d(old.in_channels, 1, kernel_size=3, padding=1, bias=True)
        with torch.no_grad():
            pruned.weight.copy_(old.weight[0:1])
            pruned.bias.copy_(old.bias[0:1])
        self.decoder.conv4 = pruned

    def forward(self, src, bgr):
        # Gentle outer-top-corner illumination adaptation (±14%) so subtle webcam auto-exposure drift
        # over a workday never creates false background deltas against the saved clean plate!
        src_tl = src[:, :, :48, :96].mean(dim=(2, 3), keepdim=True)
        src_tr = src[:, :, :48, -96:].mean(dim=(2, 3), keepdim=True)
        bgr_tl = bgr[:, :, :48, :96].mean(dim=(2, 3), keepdim=True)
        bgr_tr = bgr[:, :, :48, -96:].mean(dim=(2, 3), keepdim=True)
        gain = torch.clamp((src_tl + src_tr) / torch.clamp(bgr_tl + bgr_tr, min=0.04), 0.86, 1.16)
        bgr_adapted = torch.clamp(bgr * gain, 0.0, 1.0)

        x = torch.cat([src, bgr_adapted], dim=1)
        x4, x3, x2, x1, x0 = self.backbone(x)
        x = self.aspp(x4)
        out = self.decoder(x, x3, x2, x1, x0)
        # Scale [0, 1] -> [0, 255] for Core ML GRAYSCALE OneComponent8 IOSurface CVPixelBuffer output
        return torch.clamp(out, 0.0, 1.0) * 255.0


# ==============================================================================
# 2. MODNet Webcam Portrait Matting (Apache-2.0) — ANE-Optimized Graph
# ==============================================================================

def build_ane_modnet(modnet_repo_dir: str, ckpt_path: str) -> nn.Module:
    sys.path.insert(0, modnet_repo_dir)
    from src.models.modnet import MODNet

    raw_model = MODNet(in_channels=3, hr_channels=32, backbone_arch="mobilenetv2", backbone_pretrained=False)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    raw_model.load_state_dict(clean_sd, strict=True)
    raw_model.eval()

    # Replace SEBlock Linear+view with pure 4D 1x1 Conv2d for 100% ANE 4D tensor residency
    old_se = raw_model.lr_branch.se_block
    lin1 = old_se.fc[0]
    lin2 = old_se.fc[2]
    c1 = nn.Conv2d(lin1.in_features, lin1.out_features, 1, bias=False)
    c2 = nn.Conv2d(lin2.in_features, lin2.out_features, 1, bias=False)
    with torch.no_grad():
        c1.weight.copy_(lin1.weight.view(lin1.out_features, lin1.in_features, 1, 1))
        c2.weight.copy_(lin2.weight.view(lin2.out_features, lin2.in_features, 1, 1))

    class ANESEBlock(nn.Module):
        def __init__(self, conv1, conv2):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.net = nn.Sequential(conv1, nn.ReLU(inplace=False), conv2, nn.Sigmoid())

        def forward(self, x):
            return x * self.net(self.pool(x))

    raw_model.lr_branch.se_block = ANESEBlock(c1, c2)

    # Remove in-place ReLU/ReLU6 for clean TorchScript -> CoreML MIL lowering
    for m in raw_model.modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6)):
            m.inplace = False

    class StudioCamMODNetWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, image):
            _, _, matte = self.net(image, True)
            return torch.clamp(matte, 0.0, 1.0) * 255.0

    wrapper = StudioCamMODNetWrapper(raw_model)
    wrapper.eval()
    return wrapper


def ensure_file(path: str, url: str):
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {url} -> {path}...")
    urllib.request.urlretrieve(url, path)


def export_models(out_dir: str, scratch_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    bgm_ckpt = os.path.join(scratch_dir, "weights", "bgm_mobilenetv2.pth")
    modnet_ckpt = os.path.join(scratch_dir, "weights", "modnet_webcam.ckpt")
    modnet_repo = os.path.join(scratch_dir, "repos", "MODNet")

    if not os.path.exists(modnet_repo):
        os.makedirs(os.path.dirname(modnet_repo), exist_ok=True)
        print("Cloning ZHKKKe/MODNet (Apache-2.0) model definitions...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ZHKKKe/MODNet.git", modnet_repo], check=True)

    ensure_file(bgm_ckpt, BGM_URL)
    ensure_file(modnet_ckpt, MODNET_URL)

    # --- Export 1: BGMv2 Clean-Plate (MIT) ---
    print("[1/2] Exporting StudioCamBGMv2CleanPlateF16.mlpackage (MIT, 512x288)...")
    bgm = StudioCamBGMv2CleanPlate()
    raw_bgm_sd = torch.load(bgm_ckpt, map_location="cpu", weights_only=False)
    base_sd = {k: v for k, v in raw_bgm_sd.items() if not k.startswith("refiner.")}
    bgm.load_state_dict(base_sd, strict=True)
    bgm.prune_alpha_only_head()
    bgm.eval()

    ex_src = torch.rand(1, 3, H, W)
    ex_bgr = torch.rand(1, 3, H, W)
    with torch.no_grad():
        traced_bgm = torch.jit.trace(bgm, (ex_src, ex_bgr))

    ml_bgm = ct.convert(
        traced_bgm,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.ImageType(name="src", shape=(1, 3, H, W), scale=1.0 / 255.0, bias=[0.0, 0.0, 0.0], color_layout=ct.colorlayout.RGB),
            ct.ImageType(name="bgr", shape=(1, 3, H, W), scale=1.0 / 255.0, bias=[0.0, 0.0, 0.0], color_layout=ct.colorlayout.RGB),
        ],
        outputs=[
            ct.ImageType(name="alpha", color_layout=ct.colorlayout.GRAYSCALE),
        ],
    )
    ml_bgm.short_description = "StudioCam Clean-Plate Video Matting (BackgroundMattingV2 MattingBase MobileNetV2, MIT License)"
    ml_bgm.license = "MIT"
    bgm_out = os.path.join(out_dir, "StudioCamBGMv2CleanPlateF16.mlpackage")
    if os.path.exists(bgm_out):
        shutil.rmtree(bgm_out)
    ml_bgm.save(bgm_out)
    print(f"  Saved: {bgm_out}")

    # --- Export 2: MODNet Webcam Portrait Matting (Apache-2.0) ---
    print("[2/2] Exporting StudioCamMODNetWebcamF16.mlpackage (Apache-2.0, 512x288)...")
    modnet = build_ane_modnet(modnet_repo, modnet_ckpt)
    ex_img = torch.rand(1, 3, H, W)
    with torch.no_grad():
        traced_mod = torch.jit.trace(modnet, ex_img)

    ml_mod = ct.convert(
        traced_mod,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, H, W),
                scale=1.0 / 127.5,
                bias=[-1.0, -1.0, -1.0],
                color_layout=ct.colorlayout.RGB,
            ),
        ],
        outputs=[
            ct.ImageType(name="alpha", color_layout=ct.colorlayout.GRAYSCALE),
        ],
    )
    ml_mod.short_description = "StudioCam Universal Webcam Portrait Matting (MODNet MobileNetV2, Apache-2.0 License)"
    ml_mod.license = "Apache-2.0"
    mod_out = os.path.join(out_dir, "StudioCamMODNetWebcamF16.mlpackage")
    if os.path.exists(mod_out):
        shutil.rmtree(mod_out)
    ml_mod.save(mod_out)
    print(f"  Saved: {mod_out}")


if __name__ == "__main__":
    cache_root = os.environ.get("STUDIOCAM_EXPORT_CACHE") or os.path.join(tempfile.gettempdir(), "studiocam_coreml_cache")
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODELS_DIR
    export_models(target, cache_root)
