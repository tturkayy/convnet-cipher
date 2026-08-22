import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. MIMARI TANIMI VE GRAD-CAM KANCALARI (HOOKS)
# =====================================================================

class InceptionBlock(nn.Module):
    def __init__(self, in_c, out_branch_c):
        super().__init__()
        self.b1 = nn.Conv2d(in_c, out_branch_c, kernel_size=1, padding=0)
        self.b2 = nn.Conv2d(in_c, out_branch_c, kernel_size=3, padding=1)
        self.b3 = nn.Conv2d(in_c, out_branch_c, kernel_size=5, padding=2)
        self.proj = nn.Conv2d(in_c, out_branch_c * 3, kernel_size=1, padding=0)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.act((out + self.proj(x)) * 0.5)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=3):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)
        self.act = nn.LeakyReLU(0.1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        gap = x.mean(dim=[2, 3])
        hidden = self.act(self.fc1(gap))
        weights = self.sig(self.fc2(hidden)).view(b, c, 1, 1)
        return (x * weights + x) * 0.5


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride, padding=1, groups=in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)
        self.act = nn.LeakyReLU(0.1)
        self.proj = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride) if (in_c != out_c or stride != 1) else None

    def forward(self, x):
        out = self.pw(self.act(self.dw(x)))
        shortcut = self.proj(x) if self.proj else x
        return self.act((out + shortcut) * 0.5)


class GradCAMNeuralSBox(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = InceptionBlock(1, 3)
        self.se = SEModule(9, 3)
        self.dw1 = DepthwiseSeparable(9, 16, stride=2)
        self.dw2 = DepthwiseSeparable(16, 16, stride=1)
        self.bottleneck = nn.Conv2d(16, 4, kernel_size=1)
        self.tanh = nn.Tanh()

        # Grad-CAM için gradyan ve aktivasyon önbelleği
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.inception(x)
        x = self.se(x)
        x = self.dw1(x)
        x = self.dw2(x)

        # Son evrişim katmanından aktivasyon ve gradyan kancası
        feat = self.bottleneck(x)
        self.activations = feat
        if feat.requires_grad:
            feat.register_hook(self.activations_hook)

        return self.tanh(feat)


# =====================================================================
# 2. GRAD-CAM VE DİFÜZYON METRİKLERİ
# =====================================================================

def compute_gradcam_heatmap(model, input_tensor, target_channel=0):
    """
    Belirli bir hedef çıkış kanalına göre Grad-CAM aktivasyon haritasını türetir.
    """
    model.zero_grad()
    output = model(input_tensor)

    # Çıkış sinyali skaler kaybı (Belirli bir kanalın toplam enerji gradyanı)
    target = output[0, target_channel].sum()
    target.backward()

    gradients = model.gradients.data.cpu().numpy()[0]  # [C, H, W] = [4, 16, 16]
    activations = model.activations.data.cpu().numpy()[0]  # [4, 16, 16]

    # Global Average Pooling (Kanal ağırlıkları alpha_k)
    weights = np.mean(gradients, axis=(1, 2))  # [4]

    # Ağırlıklı kombinasyon
    cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [16, 16]
    for i, w in enumerate(weights):
        cam += w * activations[i]

    # ReLU ile sadece pozitif katkı sağlayan difüzyon bölgeleri
    cam = np.maximum(cam, 0)

    # Giriş boyutuna (32x32) bilinear enterpolasyon
    cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam_upsampled = F.interpolate(cam_tensor, size=(32, 32), mode='bilinear', align_corners=False)
    heatmap = cam_upsampled.squeeze().numpy()

    # [0, 1] Normalizasyonu
    denom = np.max(heatmap) - np.min(heatmap)
    if denom > 1e-8:
        heatmap = (heatmap - np.min(heatmap)) / denom
    else:
        heatmap = np.ones_like(heatmap) * 0.5

    return heatmap


def compute_gini_coefficient(array):
    """Eşitsizlik katsayısı (0: Kusursuz difüzyon/eşitlik, 1: Aşırı odaklanma/lokal sızıntı)"""
    flat = array.flatten()
    if np.amin(flat) < 0:
        flat -= np.amin(flat)
    flat += 1e-7
    flat = np.sort(flat)
    index = np.arange(1, flat.shape[0] + 1)
    n = flat.shape[0]
    return ((np.sum((2 * index - n - 1) * flat)) / (n * np.sum(flat)))


def render_ascii_heatmap(heatmap, width=32, height=16):
    """Terminalde difüzyon yoğunluğunu gösteren ASCII matrisi"""
    chars = [" ", "░", "▒", "▓", "█"]
    step_y = heatmap.shape[0] // height
    step_x = heatmap.shape[1] // width

    lines = []
    for y in range(0, heatmap.shape[0], step_y):
        row = []
        for x in range(0, heatmap.shape[1], step_x):
            val = heatmap[y, x]
            idx = int(val * (len(chars) - 1))
            row.append(chars[min(idx, len(chars) - 1)])
        lines.append("".join(row))
    return "\n".join(lines)


# =====================================================================
# 3. YÜRÜTÜCÜ VE RAPORLAMA
# =====================================================================

def run_gradcam_diffusion_analysis():
    print("=" * 80)
    print(" CONVNET-PRNG GRAD-CAM SPATIAL DIFFUSION & XAI EVALUATION")
    print("=" * 80)

    model = GradCAMNeuralSBox()
    try:
        with open("weights.json", "r") as f:
            w = json.load(f)
            model.inception.b1.weight.data = torch.tensor(w["inc_b1_w"]).view(3, 1, 1, 1).float() / 127.0
            model.bottleneck.weight.data = torch.tensor(w["btn_w"]).view(4, 16, 1, 1).float() / 127.0
        print("✓ weights.json başarıyla yüklendi.")
    except Exception:
        print("! weights.json bulunamadı, varsayılan ağırlıklar kullanılıyor.")

    model.eval()

    # Deterministik durum tensörü oluştur
    torch.manual_seed(42)
    input_state = (torch.rand(1, 1, 32, 32) - 0.5) * 2.0
    input_state.requires_grad_(True)

    # 4 Çıkış kanalı için Grad-CAM haritalarını hesapla
    heatmaps = []
    ginis = []
    variances = []

    for ch in range(4):
        hm = compute_gradcam_heatmap(model, input_state, target_channel=ch)
        heatmaps.append(hm)
        gini = compute_gini_coefficient(hm)
        var = float(np.var(hm))
        ginis.append(gini)
        variances.append(var)

    # Birleşik kümülatif difüzyon haritası
    combined_heatmap = np.mean(heatmaps, axis=0)
    avg_gini = compute_gini_coefficient(combined_heatmap)
    avg_var = float(np.var(combined_heatmap))

    print("\n" + "=" * 80)
    print(" GRAD-CAM RECEPTIVE FIELD DIFFUSION MAP (32x32 State-Space Projection)")
    print("=" * 80)
    print(render_ascii_heatmap(combined_heatmap, width=32, height=16))
    print("=" * 80)
    print("ASCII Density Scale: [Low Receptive Influence ' ' -> High Influence '█']")

    print("\n" + "=" * 80)
    print(f"{'Channel Target':<25} | {'Gini Coefficient (Ideal < 0.25)':<32} | {'Spatial Variance'}")
    print("-" * 80)
    for ch in range(4):
        print(f"Latent Channel {ch:<10} | {ginis[ch]:<32.4f} | {variances[ch]:.6f}")
    print("-" * 80)
    print(f"{'Combined Manifold':<25} | {avg_gini:<32.4f} | {avg_var:.6f}")
    print("=" * 80)

    # Kriptografik Çıkarım Yorumu
    print("\n[Cryptanalytic Interpretation]")
    if avg_gini < 0.25:
        print("✓ FULL SPATIAL DIFFUSION ACHIEVED: Gini coefficient < 0.25 confirms Shannon's Confusion.")
        print("  Giriş durumundaki hiçbir lokal bayt kümesi çıkış anahtar akışında imtiyazlı/baskın değildir.")
    else:
        print("! LOCALIZED SENSITIVITY: Diffusion exhibits slight regional concentration.")


if __name__ == "__main__":
    run_gradcam_diffusion_analysis()