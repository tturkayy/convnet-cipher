import json
import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# 1. MIMARI TANIMI (PyTorch S-Box)
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


class NeuralSBox(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = InceptionBlock(1, 3)
        self.se = SEModule(9, 3)
        self.dw1 = DepthwiseSeparable(9, 16, stride=2)
        self.dw2 = DepthwiseSeparable(16, 16, stride=1)
        self.bottleneck = nn.Conv2d(16, 4, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.inception(x)
        x = self.se(x)
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.bottleneck(x)
        return self.tanh(x)


def whiten_byte(val, idx=0):
    x = (int(val) ^ ((idx * 0x517cc1b7) & 0xFFFFFFFF)) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    x = (x * 0x85ebca6b) & 0xFFFFFFFF
    x = (x ^ (x >> 13)) & 0xFFFFFFFF
    x = (x * 0xc2b2ae35) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    return x & 0xFF


# =====================================================================
# 2. BOOLEAN VE S-BOX MATEMATIKSEL ANALIZ FONKSIYONLARI
# =====================================================================

def fast_walsh_hadamard_transform(f):
    """f: +/-1 dizisi (uzunluk 2^n). FWHT donusumu dondurur."""
    had = np.array(f, dtype=np.int64)
    n = len(had)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = had[j]
                y = had[j + h]
                had[j] = x + y
                had[j + h] = x - y
        h *= 2
    return had


def compute_sbox_nonlinearity(sbox_table):
    """
    8x8 S-Box icin her lineer bilesen fonksiyonunun (255 non-zero mask)
    Walsh spektrumunu ve minimum Nonlinearity (NL) degerini hesaplar.
    """
    n = 8
    num_inputs = 256
    min_nl = float('inf')
    component_nls = []

    for b in range(1, 256):
        # bilesen fonksiyonu f(x) = b . S(x) (GF(2) ic carpimi)
        f = np.zeros(num_inputs, dtype=np.int64)
        for x in range(num_inputs):
            out = sbox_table[x]
            parity = bin(out & b).count('1') % 2
            f[x] = 1 - 2 * parity  # 0 -> +1, 1 -> -1

        wht = fast_walsh_hadamard_transform(f)
        max_walsh = np.max(np.abs(wht))
        nl = (2 ** (n - 1)) - (max_walsh // 2)
        component_nls.append(nl)
        if nl < min_nl:
            min_nl = nl

    return min_nl, float(np.mean(component_nls))


def compute_ddt_and_mdp(sbox_table):
    """
    Fark Dagilim Tablosu (Difference Distribution Table - DDT)
    ve Maksimum Fark Olasiligi (Maximum Difference Probability - MDP)
    """
    ddt = np.zeros((256, 256), dtype=np.int32)
    for dx in range(256):
        for x in range(256):
            x_prime = x ^ dx
            dy = sbox_table[x] ^ sbox_table[x_prime]
            ddt[dx, dy] += 1

    # dx = 0 haric max deger
    max_entry = np.max(ddt[1:, :])
    mdp = max_entry / 256.0
    return ddt, max_entry, mdp


def compute_algebraic_degree(sbox_table):
    """
    Moebius / Fast Reed-Muller Donusumu ile 8 bilesen biti icin
    Cebirsel Normal Form (ANF) derecesini hesaplar.
    """

    def get_degree_of_bit(truth_table):
        a = list(truth_table)
        n = 8
        for i in range(n):
            for j in range(2 ** n):
                if (j & (1 << i)) != 0:
                    a[j] = a[j] ^ a[j ^ (1 << i)]

        max_deg = 0
        for i in range(2 ** n):
            if a[i] == 1:
                deg = bin(i).count('1')
                if deg > max_deg:
                    max_deg = deg
        return max_deg

    degrees = []
    for bit_idx in range(8):
        tt = [(sbox_table[x] >> bit_idx) & 1 for x in range(256)]
        deg = get_degree_of_bit(tt)
        degrees.append(deg)

    return min(degrees), max(degrees), degrees


# =====================================================================
# 3. YÜRÜTÜCÜ VE RAPORLAMA
# =====================================================================

def evaluate_boolean_cryptanalysis():
    print("=" * 80)
    print(" BOOLEAN FUNCTION & S-BOX CRYPTANALYSIS PIPELINE")
    print("=" * 80)

    # 1. AES S-Box Referansı
    AES_SBOX = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
    ]

    # 2. Modeli Yükle
    model = NeuralSBox()
    try:
        with open("weights.json", "r") as f:
            w = json.load(f)
            model.inception.b1.weight.data = torch.tensor(w["inc_b1_w"]).view(3, 1, 1, 1).float() / 127.0
            model.bottleneck.weight.data = torch.tensor(w["btn_w"]).view(4, 16, 1, 1).float() / 127.0
        print("✓ weights.json basariyla yuklendi.")
    except Exception:
        print("! weights.json bulunamadi, varsayilan agirliklar kullaniliyor.")
    model.eval()

    # 3. Modelden 256 Elemanlik Deterministik S-Box Uret
    neural_sbox = []
    with torch.no_grad():
        for byte_val in range(256):
            state = np.full((1, 1, 32, 32), (byte_val - 128) / 128.0, dtype=np.float32)
            out = model(torch.tensor(state)).numpy().flatten()
            q_byte = int(out[0] * 127.0)
            neural_sbox.append(whiten_byte(q_byte, byte_val))

    # Rastgele S-Box (Karsilastirma)
    np.random.seed(42)
    random_sbox = list(np.random.permutation(256))

    targets = {
        "1. Standard AES-256 S-Box (Optimal)": AES_SBOX,
        "2. Random Permutation S-Box": random_sbox,
        "3. ConvNet Neural S-Box (Ours)": neural_sbox
    }

    results = []
    for name, sbox in targets.items():
        min_nl, avg_nl = compute_sbox_nonlinearity(sbox)
        ddt, max_diff, mdp = compute_ddt_and_mdp(sbox)
        min_deg, max_deg, bit_degs = compute_algebraic_degree(sbox)

        results.append({
            "name": name,
            "min_nl": min_nl,
            "avg_nl": avg_nl,
            "max_ddt": max_diff,
            "mdp": mdp,
            "deg": min_deg
        })

    print("\n" + "=" * 90)
    print(f"{'S-Box Target':<38} | {'Min NL':<8} | {'Avg NL':<8} | {'Max DDT':<8} | {'MDP':<8} | {'Degree'}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['name']:<38} | {r['min_nl']:<8} | {r['avg_nl']:<8.2f} | {r['max_ddt']:<8} | {r['mdp']:<8.4f} | {r['deg']}")
    print("=" * 90)


if __name__ == "__main__":
    evaluate_boolean_cryptanalysis()