import numpy as np
import torch
import torch.nn as nn
from scipy import stats


# =====================================================================
# 1. YARDIMCI VE KRİPTOGRAFİK WHITENING
# =====================================================================

def whiten_byte(val, idx=0):
    x = (int(val) ^ ((idx * 0x517cc1b7) & 0xFFFFFFFF)) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    x = (x * 0x85ebca6b) & 0xFFFFFFFF
    x = (x ^ (x >> 13)) & 0xFFFFFFFF
    x = (x * 0xc2b2ae35) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    return x & 0xFF


def calculate_entropy(byte_array):
    counts = np.bincount(byte_array, minlength=256)
    probs = counts[counts > 0] / len(byte_array)
    return -np.sum(probs * np.log2(probs))


def init_weights(m):
    """Kriptografik kaos ve difuzyon icin agirliklari olceklendir"""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=1.2)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -0.5, 0.5)


# =====================================================================
# 2. MIMARI VARYASYONLARI
# =====================================================================

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Conv2d(1, 3, 1)
        self.b2 = nn.Conv2d(1, 3, 3, padding=1)
        self.b3 = nn.Conv2d(1, 3, 5, padding=2)
        self.proj1 = nn.Conv2d(1, 9, 1)

        self.fc1 = nn.Linear(9, 3)
        self.fc2 = nn.Linear(3, 9)

        self.dw1 = nn.Conv2d(9, 9, 3, stride=2, padding=1, groups=9)
        self.pw1 = nn.Conv2d(9, 16, 1)
        self.proj2 = nn.Conv2d(9, 16, 1, stride=2)

        self.dw2 = nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=16)
        self.pw2 = nn.Conv2d(16, 16, 1)

        self.btn = nn.Conv2d(16, 4, 1)
        self.act = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()
        self.apply(init_weights)

    def forward(self, x):
        inc = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        x_inc = self.act((inc + self.proj1(x)) * 0.5)

        b, c, _, _ = x_inc.size()
        gap = x_inc.mean(dim=[2, 3])
        w = torch.sigmoid(self.fc2(self.act(self.fc1(gap)))).view(b, c, 1, 1)
        x_se = (x_inc * w + x_inc) * 0.5

        dw1 = self.pw1(self.act(self.dw1(x_se)))
        x_dw1 = self.act((dw1 + self.proj2(x_se)) * 0.5)

        dw2 = self.pw2(self.act(self.dw2(x_dw1)))
        x_dw2 = self.act((dw2 + x_dw1) * 0.5)

        return self.tanh(self.btn(x_dw2) * 2.0)


class NoSEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Conv2d(1, 3, 1)
        self.b2 = nn.Conv2d(1, 3, 3, padding=1)
        self.b3 = nn.Conv2d(1, 3, 5, padding=2)
        self.proj1 = nn.Conv2d(1, 9, 1)

        self.dw1 = nn.Conv2d(9, 9, 3, stride=2, padding=1, groups=9)
        self.pw1 = nn.Conv2d(9, 16, 1)
        self.proj2 = nn.Conv2d(9, 16, 1, stride=2)

        self.dw2 = nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=16)
        self.pw2 = nn.Conv2d(16, 16, 1)

        self.btn = nn.Conv2d(16, 4, 1)
        self.act = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()
        self.apply(init_weights)

    def forward(self, x):
        inc = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        x_inc = self.act((inc + self.proj1(x)) * 0.5)

        dw1 = self.pw1(self.act(self.dw1(x_inc)))
        x_dw1 = self.act((dw1 + self.proj2(x_inc)) * 0.5)

        dw2 = self.pw2(self.act(self.dw2(x_dw1)))
        x_dw2 = self.act((dw2 + x_dw1) * 0.5)

        return self.tanh(self.btn(x_dw2) * 2.0)


class No5x5Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Conv2d(1, 4, 1)
        self.b2 = nn.Conv2d(1, 4, 3, padding=1)
        self.proj1 = nn.Conv2d(1, 8, 1)

        self.fc1 = nn.Linear(8, 2)
        self.fc2 = nn.Linear(2, 8)

        self.dw1 = nn.Conv2d(8, 8, 3, stride=2, padding=1, groups=8)
        self.pw1 = nn.Conv2d(8, 16, 1)
        self.proj2 = nn.Conv2d(8, 16, 1, stride=2)

        self.dw2 = nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=16)
        self.pw2 = nn.Conv2d(16, 16, 1)

        self.btn = nn.Conv2d(16, 4, 1)
        self.act = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()
        self.apply(init_weights)

    def forward(self, x):
        inc = torch.cat([self.b1(x), self.b2(x)], dim=1)
        x_inc = self.act((inc + self.proj1(x)) * 0.5)

        b, c, _, _ = x_inc.size()
        gap = x_inc.mean(dim=[2, 3])
        w = torch.sigmoid(self.fc2(self.act(self.fc1(gap)))).view(b, c, 1, 1)
        x_se = (x_inc * w + x_inc) * 0.5

        dw1 = self.pw1(self.act(self.dw1(x_se)))
        x_dw1 = self.act((dw1 + self.proj2(x_se)) * 0.5)

        dw2 = self.pw2(self.act(self.dw2(x_dw1)))
        x_dw2 = self.act((dw2 + x_dw1) * 0.5)

        return self.tanh(self.btn(x_dw2) * 2.0)


class NoShortcutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Conv2d(1, 3, 1)
        self.b2 = nn.Conv2d(1, 3, 3, padding=1)
        self.b3 = nn.Conv2d(1, 3, 5, padding=2)

        self.fc1 = nn.Linear(9, 3)
        self.fc2 = nn.Linear(3, 9)

        self.dw1 = nn.Conv2d(9, 9, 3, stride=2, padding=1, groups=9)
        self.pw1 = nn.Conv2d(9, 16, 1)

        self.dw2 = nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=16)
        self.pw2 = nn.Conv2d(16, 16, 1)

        self.btn = nn.Conv2d(16, 4, 1)
        self.act = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()
        self.apply(init_weights)

    def forward(self, x):
        inc = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        x_inc = self.act(inc)

        b, c, _, _ = x_inc.size()
        gap = x_inc.mean(dim=[2, 3])
        w = torch.sigmoid(self.fc2(self.act(self.fc1(gap)))).view(b, c, 1, 1)
        x_se = x_inc * w

        x_dw1 = self.act(self.pw1(self.act(self.dw1(x_se))))
        x_dw2 = self.act(self.pw2(self.act(self.dw2(x_dw1))))

        return self.tanh(self.btn(x_dw2) * 2.0)


# =====================================================================
# 3. METRİK VE MONTE CARLO YÜRÜTÜCÜSÜ
# =====================================================================

def evaluate_model_sample(model, rounds=3):
    with torch.no_grad():
        raw1 = np.random.randint(-128, 128, size=(32, 32), dtype=np.int32)
        raw2 = raw1.copy()

        # 1-bit diferansiyel
        ri, rj = np.random.randint(0, 32), np.random.randint(0, 32)
        raw2[ri, rj] ^= 0x01

        s1 = torch.tensor(raw1, dtype=torch.float32).view(1, 1, 32, 32) / 128.0
        s2 = torch.tensor(raw2, dtype=torch.float32).view(1, 1, 32, 32) / 128.0

        for _ in range(rounds):
            s1 = model(s1).view(1, 1, 32, 32)
            s2 = model(s2).view(1, 1, 32, 32)

        # Surekli uzaydaki non-lineer bit projeksiyonu
        bits1 = (s1.numpy().flatten() > 0.0).astype(np.uint8)
        bits2 = (s2.numpy().flatten() > 0.0).astype(np.uint8)

        # SAC (Bit Inversion Rate)
        flips = np.sum(bits1 ^ bits2)
        sac = flips / len(bits1)

        # Byte donusumu ve Entropi
        q1 = np.clip((s1.numpy().flatten() + 1.0) * 127.5, 0, 255).astype(np.uint8)
        y1 = np.array([whiten_byte(q1[i], i) for i in range(len(q1))], dtype=np.uint8)
        ent = calculate_entropy(y1)

        corr = float(np.corrcoef(raw1.flatten(), s1.numpy().flatten())[0, 1])

    return sac, ent, abs(corr)


def run_ablation_study(num_runs=30):
    print("=" * 85)
    print(f" CONVNET-PRNG ARCHITECTURAL ABLATION STUDY (N={num_runs} Monte Carlo Runs)")
    print("=" * 85)

    variants = {
        "Full Architecture (Baseline)": FullModel(),
        "w/o Squeeze-and-Excitation (SE)": NoSEModel(),
        "w/o Inception 5x5 Branch": No5x5Model(),
        "w/o Residual Shortcuts": NoShortcutModel()
    }

    metrics = {name: {"sac": [], "ent": [], "corr": []} for name in variants}

    for name, model in variants.items():
        model.eval()
        print(f"[*] Analiz ediliyor: {name}...")
        for _ in range(num_runs):
            sac, ent, corr = evaluate_model_sample(model)
            metrics[name]["sac"].append(sac)
            metrics[name]["ent"].append(ent)
            metrics[name]["corr"].append(corr)

    # Ozet Tablosu
    print("\n" + "=" * 90)
    print(f"{'Variant':<35} | {'SAC (Mean ± Std)':<18} | {'Entropy (b/b)':<16} | {'|Corr| (Mean ± Std)'}")
    print("-" * 90)
    for name in variants:
        s_m, s_sd = np.mean(metrics[name]["sac"]), np.std(metrics[name]["sac"])
        e_m, e_sd = np.mean(metrics[name]["ent"]), np.std(metrics[name]["ent"])
        c_m, c_sd = np.mean(metrics[name]["corr"]), np.std(metrics[name]["corr"])
        print(f"{name:<35} | {s_m:.4f} ± {s_sd:.4f}     | {e_m:.4f} ± {e_sd:.4f}   | {c_m:.4f} ± {c_sd:.4f}")
    print("=" * 90)

    # ANOVA ve T-Test Analizi
    print("\n" + "=" * 85)
    print(" STATISTICAL SIGNIFICANCE EVALUATION (One-Way ANOVA & Welch's T-Test)")
    print("=" * 85)

    base_sac = metrics["Full Architecture (Baseline)"]["sac"]
    all_sac_groups = [metrics[name]["sac"] for name in variants]

    f_stat, p_val_anova = stats.f_oneway(*all_sac_groups)
    print(f"One-Way ANOVA across all variants: F-Score = {f_stat:.4f}, p-value = {p_val_anova:.4e}")

    print("\nPairwise Comparisons vs. Full Architecture (Welch's t-test):")
    for name in list(variants.keys())[1:]:
        t_stat, p_val_t = stats.ttest_ind(base_sac, metrics[name]["sac"], equal_var=False)
        sig = "Significant (p < 0.05)" if p_val_t < 0.05 else "Not Significant (p >= 0.05)"
        print(f"  * {name:<32} -> t = {t_stat:+.4f}, p = {p_val_t:.4e} [{sig}]")
    print("=" * 85)


if __name__ == "__main__":
    run_ablation_study(num_runs=30)