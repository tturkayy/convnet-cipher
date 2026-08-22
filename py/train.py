import json
import torch
import torch.nn as nn
import torch.optim as optim


class InceptionBlockPyTorch(nn.Module):
    def __init__(self, in_c, out_branch_c):
        super().__init__()
        self.b1 = nn.Conv2d(in_c, out_branch_c, kernel_size=1, padding=0)
        self.b2 = nn.Conv2d(in_c, out_branch_c, kernel_size=3, padding=1)
        self.b3 = nn.Conv2d(in_c, out_branch_c, kernel_size=5, padding=2)
        self.proj = nn.Conv2d(in_c, out_branch_c * 3, kernel_size=1, padding=0)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        shortcut = self.proj(x)
        return self.act((out + shortcut) * 0.5)


class SEModulePyTorch(nn.Module):
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
        scaled = x * weights
        return (scaled + x) * 0.5


class DepthwiseSeparablePyTorch(nn.Module):
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


class NeuralSBoxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = InceptionBlockPyTorch(1, 3)
        self.se = SEModulePyTorch(9, 3)
        self.dw1 = DepthwiseSeparablePyTorch(9, 16, stride=2)
        self.dw2 = DepthwiseSeparablePyTorch(16, 16, stride=1)
        self.bottleneck = nn.Conv2d(16, 4, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.inception(x)
        x = self.se(x)
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.bottleneck(x)
        return self.tanh(x)


def cryptanalytic_loss(output):
    # Çıktıyı [-1, 1] aralığından [0, 1] aralığına normalize et
    norm_y = (output.view(-1) + 1.0) * 0.5

    # 1. Uniformity Loss (Histogram Düzgünlüğü - CDF Farkı)
    sorted_y, _ = torch.sort(norm_y)
    n = sorted_y.size(0)
    ideal_cdf = torch.linspace(0, 1, steps=n, device=output.device)
    loss_uniform = torch.mean((sorted_y - ideal_cdf) ** 2)

    # 2. Autocorrelation Loss (Lag-1 İlişkisizliği)
    y_centered = norm_y - torch.mean(norm_y)
    variance = torch.sum(y_centered ** 2) + 1e-8
    autocorr = torch.sum(y_centered[:-1] * y_centered[1:]) / variance
    loss_autocorr = autocorr ** 2

    return loss_uniform + 0.5 * loss_autocorr


def quantize_to_int(tensor, max_range=127):
    # Float ağırlıkları [-max_range, max_range] tamsayılarına nicemler
    clamped = torch.clamp(tensor, -1.0, 1.0)
    return (clamped * max_range).round().int().tolist()


def train_and_export():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralSBoxModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    print("Model eğitimi başladı...")
    for epoch in range(1, 501):
        optimizer.zero_grad()
        # Giriş: Simüle edilmiş AES durum tensörleri [-1, 1]
        dummy_state = (torch.rand(32, 1, 32, 32, device=device) - 0.5) * 2.0
        out = model(dummy_state)

        loss = cryptanalytic_loss(out)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/500 - Loss: {loss.item():.6f}")

    print("Eğitim tamamlandı. Ağırlıklar quantize edilip JSON olarak dışa aktarılıyor...")

    # Model katmanlarını JavaScript'in doğrudan okuyacağı yapıya çıkar
    weights_dict = {
        # Inception
        "inc_b1_w": quantize_to_int(model.inception.b1.weight.data.view(-1)),
        "inc_b1_b": quantize_to_int(model.inception.b1.bias.data.view(-1), 63),
        "inc_b2_w": quantize_to_int(model.inception.b2.weight.data.view(-1)),
        "inc_b2_b": quantize_to_int(model.inception.b2.bias.data.view(-1), 63),
        "inc_b3_w": quantize_to_int(model.inception.b3.weight.data.view(-1)),
        "inc_b3_b": quantize_to_int(model.inception.b3.bias.data.view(-1), 63),
        "inc_proj_w": quantize_to_int(model.inception.proj.weight.data.view(-1)),
        "inc_proj_b": quantize_to_int(model.inception.proj.bias.data.view(-1), 63),

        # SE Module
        "se_fc1_w": quantize_to_int(model.se.fc1.weight.data.view(-1), 63),
        "se_fc2_w": quantize_to_int(model.se.fc2.weight.data.view(-1), 63),

        # DW1
        "dw1_dw_w": quantize_to_int(model.dw1.dw.weight.data.view(-1)),
        "dw1_dw_b": quantize_to_int(model.dw1.dw.bias.data.view(-1), 63),
        "dw1_pw_w": quantize_to_int(model.dw1.pw.weight.data.view(-1)),
        "dw1_pw_b": quantize_to_int(model.dw1.pw.bias.data.view(-1), 63),
        "dw1_proj_w": quantize_to_int(model.dw1.proj.weight.data.view(-1)),
        "dw1_proj_b": quantize_to_int(model.dw1.proj.bias.data.view(-1), 63),

        # DW2
        "dw2_dw_w": quantize_to_int(model.dw2.dw.weight.data.view(-1)),
        "dw2_dw_b": quantize_to_int(model.dw2.dw.bias.data.view(-1), 63),
        "dw2_pw_w": quantize_to_int(model.dw2.pw.weight.data.view(-1)),
        "dw2_pw_b": quantize_to_int(model.dw2.pw.bias.data.view(-1), 63),

        # Bottleneck
        "btn_w": quantize_to_int(model.bottleneck.weight.data.view(-1)),
        "btn_b": quantize_to_int(model.bottleneck.bias.data.view(-1), 63)
    }

    with open("weights.json", "w") as f:
        json.dump(weights_dict, f)
    print("✓ weights.json başarıyla oluşturuldu.")


if __name__ == "__main__":
    train_and_export()