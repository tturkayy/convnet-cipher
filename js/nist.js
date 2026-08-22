function erfc(x) {
    const z = Math.abs(x);
    const t = 1.0 / (1.0 + 0.5 * z);
    const ans = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? ans : 2.0 - ans;
}

function normalCdf(x) {
    return 0.5 * erfc(-x / Math.SQRT2);
}

function gammp(a, x) {
    if (x < 0 || a <= 0) return 0;
    if (x === 0) return 0;
    let sum = 1.0 / a, term = sum;
    for (let n = 1; n < 100; n++) {
        term *= x / (a + n);
        sum += term;
        if (Math.abs(term) < Math.abs(sum) * 1e-10) break;
    }
    return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
}

function logGamma(z) {
    const c = [57.1562356658629235, -59.5979603554754912, 14.1360979740577471, -0.491913816097620199, .339946499848118887e-4, .465236289270485756e-4, -.983744753048795646e-4, .158088703224377488e-3, -.210264441724104783e-3, .217439618115212643e-3, -.164318106536763890e-3, .844182239838527433e-4, -.261908384015814087e-4, .368991826595316227e-5];
    let y = z, x = z, tmp = x + 5.24218750000000000;
    tmp = (x + 0.5) * Math.log(tmp) - tmp;
    let ser = 0.99999999999999709182;
    for (let i = 0; i < 14; i++) ser += c[i] / ++y;
    return tmp + Math.log(2.5066282746310005 * ser / x);
}

// 1. Monobit Frequency
function runNistMonobit(bytes) {
    let s_n = 0;
    const n = bytes.length * 8;
    for (let i = 0; i < bytes.length; i++) {
        let b = bytes[i];
        for (let j = 0; j < 8; j++) {
            s_n += (b & 1) ? 1 : -1;
            b >>= 1;
        }
    }
    const s_obs = Math.abs(s_n) / Math.sqrt(n);
    const p_value = erfc(s_obs / Math.SQRT2);
    return { s_obs, p_value, pass: p_value >= 0.01 };
}

// 2. Block Frequency (m = 128)
function runNistBlockFrequency(bytes, m = 128) {
    const n = bytes.length * 8;
    const N = Math.floor(n / m);
    if (N === 0) return { chi2: 0, p_value: 1.0, pass: true };

    const bits = new Uint8Array(n);
    let idx = 0;
    for (let i = 0; i < bytes.length; i++) {
        let b = bytes[i];
        for (let j = 0; j < 8; j++) {
            bits[idx++] = (b & 1);
            b >>= 1;
        }
    }

    let chi2 = 0.0;
    for (let i = 0; i < N; i++) {
        let sum = 0;
        for (let j = 0; j < m; j++) sum += bits[i * m + j];
        const pi = sum / m;
        chi2 += (pi - 0.5) * (pi - 0.5);
    }
    chi2 *= 4.0 * m;
    const p_value = Math.max(0.0, Math.min(1.0, 1.0 - gammp(N / 2.0, chi2 / 2.0)));
    return { chi2, p_value, pass: p_value >= 0.01 };
}

// 3. Cumulative Sums (Cusum Forward)
function runNistCusum(bytes) {
    const n = bytes.length * 8;
    let s = 0, maxZ = 0;
    for (let i = 0; i < bytes.length; i++) {
        let b = bytes[i];
        for (let j = 0; j < 8; j++) {
            s += (b & 1) ? 1 : -1;
            if (Math.abs(s) > maxZ) maxZ = Math.abs(s);
            b >>= 1;
        }
    }
    const z = maxZ;
    let sum1 = 0, sum2 = 0;
    const sqN = Math.sqrt(n);
    for (let k = Math.floor((-n / z + 1) / 4); k <= Math.floor((n / z - 1) / 4); k++) {
        sum1 += normalCdf(((4 * k + 1) * z) / sqN) - normalCdf(((4 * k - 1) * z) / sqN);
    }
    for (let k = Math.floor((-n / z - 3) / 4); k <= Math.floor((n / z - 1) / 4); k++) {
        sum2 += normalCdf(((4 * k + 3) * z) / sqN) - normalCdf(((4 * k + 1) * z) / sqN);
    }
    const p_value = Math.max(0.0, Math.min(1.0, 1.0 - sum1 + sum2));
    return { z, p_value, pass: p_value >= 0.01 };
}

// 4. Runs Test
function runNistRuns(bytes) {
    const n = bytes.length * 8;
    let ones = 0;
    const bits = new Uint8Array(n);
    let idx = 0;
    for (let i = 0; i < bytes.length; i++) {
        let b = bytes[i];
        for (let j = 0; j < 8; j++) {
            const bit = (b & 1);
            bits[idx++] = bit;
            if (bit === 1) ones++;
            b >>= 1;
        }
    }
    const pi = ones / n;
    if (Math.abs(pi - 0.5) >= (2.0 / Math.sqrt(n))) return { v_n: 0, p_value: 0.0, pass: false };
    let v_n = 1;
    for (let k = 0; k < n - 1; k++) {
        if (bits[k] !== bits[k + 1]) v_n++;
    }
    const num = Math.abs(v_n - 2.0 * n * pi * (1.0 - pi));
    const den = 2.0 * Math.sqrt(2.0 * n) * pi * (1.0 - pi);
    const p_value = erfc(num / den);
    return { v_n, p_value, pass: p_value >= 0.01 };
}

// 5. Discrete Fourier Transform (Spectral)
function fft(real, imag) {
    const n = real.length;
    let j = 0;
    for (let i = 0; i < n - 1; i++) {
        if (i < j) {
            let tr = real[i], ti = imag[i];
            real[i] = real[j]; imag[i] = imag[j];
            real[j] = tr; imag[j] = ti;
        }
        let k = n >> 1;
        while (k <= j) { j -= k; k >>= 1; }
        j += k;
    }
    for (let l = 2; l <= n; l <<= 1) {
        const ang = -2 * Math.PI / l;
        const wstepR = Math.cos(ang), wstepI = Math.sin(ang);
        for (let i = 0; i < n; i += l) {
            let curR = 1, curI = 0;
            for (let k = 0; k < (l >> 1); k++) {
                const pos = i + k + (l >> 1);
                const uR = real[i + k], uI = imag[i + k];
                const vR = real[pos] * curR - imag[pos] * curI;
                const vI = real[pos] * curI + imag[pos] * curR;
                real[i + k] = uR + vR; imag[i + k] = uI + vI;
                real[pos] = uR - vR; imag[pos] = uI - vI;
                const nextR = curR * wstepR - curI * wstepI;
                curI = curR * wstepI + curI * wstepR;
                curR = nextR;
            }
        }
    }
}

function runNistDFT(bytes) {
    const totalBits = bytes.length * 8;
    let n = 1;
    while ((n << 1) <= totalBits && (n << 1) <= 16384) n <<= 1;

    const real = new Float64Array(n);
    const imag = new Float64Array(n);
    let idx = 0;
    for (let i = 0; i < bytes.length && idx < n; i++) {
        let b = bytes[i];
        for (let j = 0; j < 8 && idx < n; j++) {
            real[idx++] = (b & 1) ? 1.0 : -1.0;
            b >>= 1;
        }
    }

    fft(real, imag);

    const half = n >> 1;
    const threshold = Math.sqrt(2.995732273553991 * n);
    const n0 = 0.95 * half;
    let n1 = 0;

    for (let i = 0; i < half; i++) {
        const mag = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
        if (mag < threshold) n1++;
    }

    const d = (n1 - n0) / Math.sqrt((n * 0.95 * 0.05) / 4.0);
    const p_value = erfc(Math.abs(d) / Math.SQRT2);
    return { d, p_value, pass: p_value >= 0.01 };
}

// 6. Non-Overlapping Template Matching (m=9, B=000000001)
function runNistNonOverlappingTemplate(bytes, m = 9) {
    const n = bytes.length * 8;
    const N = 8;
    const M = Math.floor(n / N);
    if (M < m) return { chi2: 0, p_value: 1.0, pass: true };

    const bits = new Uint8Array(n);
    let idx = 0;
    for (let i = 0; i < bytes.length; i++) {
        let b = bytes[i];
        for (let j = 0; j < 8; j++) {
            bits[idx++] = (b & 1);
            b >>= 1;
        }
    }

    const lambda = (M - m + 1) / Math.pow(2, m);
    const varW = M * (1.0 / Math.pow(2, m) - (2.0 * m - 1) / Math.pow(2, 2 * m));

    let chi2 = 0;
    for (let i = 0; i < N; i++) {
        let count = 0;
        let pos = i * M;
        while (pos <= (i + 1) * M - m) {
            let match = true;
            for (let k = 0; k < m - 1; k++) {
                if (bits[pos + k] !== 0) { match = false; break; }
            }
            if (match && bits[pos + m - 1] === 1) {
                count++;
                pos += m;
            } else {
                pos++;
            }
        }
        chi2 += Math.pow(count - lambda, 2) / varW;
    }
    const p_value = Math.max(0.0, Math.min(1.0, 1.0 - gammp(N / 2.0, chi2 / 2.0)));
    return { chi2, p_value, pass: p_value >= 0.01 };
}

// 7. Approximate Entropy (ApEn, m = 2)
function runNistApproximateEntropy(bytes, m = 2) {
    const n = bytes.length * 8;
    const bits = new Uint8Array(n);
    let idx = 0;
    for (let i = 0; i < bytes.length; i++) {
        let b = bytes[i];
        for (let j = 0; j < 8; j++) {
            bits[idx++] = (b & 1);
            b >>= 1;
        }
    }

    function computePhi(k) {
        const numBlocks = 1 << k;
        const counts = new Uint32Array(numBlocks);
        for (let i = 0; i < n; i++) {
            let blockVal = 0;
            for (let j = 0; j < k; j++) blockVal = (blockVal << 1) | bits[(i + j) % n];
            counts[blockVal]++;
        }
        let sum = 0.0;
        for (let i = 0; i < numBlocks; i++) {
            if (counts[i] > 0) {
                const p = counts[i] / n;
                sum += p * Math.log(p);
            }
        }
        return sum;
    }

    const phi_m = computePhi(m);
    const phi_m1 = computePhi(m + 1);
    const apen = phi_m - phi_m1;
    const chi2 = 2.0 * n * (Math.LN2 - apen);
    const p_value = Math.max(0.0, Math.min(1.0, Math.exp(-chi2 / 2.0)));
    return { apen, p_value, pass: p_value >= 0.01 };
}

// 8. Serial Test (m = 2)
function runNistSerial(bytes, m = 2) {
    const n = bytes.length * 8;
    const bits = new Uint8Array(n);
    let idx = 0;
    for (let i = 0; i < bytes.length; i++) {
        let b = bytes[i];
        for (let j = 0; j < 8; j++) {
            bits[idx++] = (b & 1);
            b >>= 1;
        }
    }

    function psiSquare(k) {
        if (k === 0) return 0;
        const numBlocks = 1 << k;
        const counts = new Uint32Array(numBlocks);
        for (let i = 0; i < n; i++) {
            let blockVal = 0;
            for (let j = 0; j < k; j++) blockVal = (blockVal << 1) | bits[(i + j) % n];
            counts[blockVal]++;
        }
        let sum = 0.0;
        for (let i = 0; i < numBlocks; i++) sum += counts[i] * counts[i];
        return (numBlocks / n) * sum - n;
    }

    const psi2_m = psiSquare(m);
    const psi2_m1 = psiSquare(m - 1);
    const del1 = psi2_m - psi2_m1;
    const p_value = Math.max(0.0, Math.min(1.0, 1.0 - gammp((1 << (m - 2)), del1 / 2.0)));
    return { del1, p_value, pass: p_value >= 0.01 };
}

// 9. Linear Complexity (Berlekamp-Massey Algorithm, M=500)
function runNistLinearComplexity(bytes, M = 500) {
    const n = bytes.length * 8;
    const N = Math.floor(n / M);
    if (N === 0) return { chi2: 0, p_value: 1.0, pass: true };

    const bits = new Uint8Array(n);
    let idx = 0;
    for (let i = 0; i < bytes.length; i++) {
        let b = bytes[i];
        for (let j = 0; j < 8; j++) {
            bits[idx++] = (b & 1);
            b >>= 1;
        }
    }

    const mu = M / 2.0 + (9.0 + (M % 2 === 0 ? 1.0 : -1.0)) / 36.0 - (M / 3.0 + 2.0 / 9.0) / Math.pow(2, M);
    const nu = [0, 0, 0, 0, 0, 0, 0];
    const pi = [0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.02078];

    for (let i = 0; i < N; i++) {
        // Berlekamp-Massey for block i
        const block = bits.subarray(i * M, (i + 1) * M);
        let B = new Uint8Array(M);
        let C = new Uint8Array(M);
        B[0] = 1; C[0] = 1;
        let L = 0, m = -1;

        for (let N_idx = 0; N_idx < M; N_idx++) {
            let d = block[N_idx];
            for (let j = 1; j <= L; j++) d ^= (C[j] & block[N_idx - j]);
            if (d === 1) {
                let T = new Uint8Array(C);
                let shift = N_idx - m;
                for (let j = 0; j + shift < M; j++) C[j + shift] ^= B[j];
                if (L <= N_idx / 2) {
                    L = N_idx + 1 - L;
                    m = N_idx;
                    B = T;
                }
            }
        }

        const T_val = (M % 2 === 0 ? 1 : -1) * (L - mu) + 2.0 / 9.0;
        if (T_val <= -2.5) nu[0]++;
        else if (T_val <= -1.5) nu[1]++;
        else if (T_val <= -0.5) nu[2]++;
        else if (T_val <= 0.5) nu[3]++;
        else if (T_val <= 1.5) nu[4]++;
        else if (T_val <= 2.5) nu[5]++;
        else nu[6]++;
    }

    let chi2 = 0.0;
    for (let k = 0; k < 7; k++) {
        chi2 += Math.pow(nu[k] - N * pi[k], 2) / (N * pi[k]);
    }
    const p_value = Math.max(0.0, Math.min(1.0, 1.0 - gammp(3.0, chi2 / 2.0)));
    return { chi2, p_value, pass: p_value >= 0.01 };
}

// 10. Chi-Square Goodness-of-Fit (df = 255)
function runChiSquare(bytes) {
    const counts = new Uint32Array(256);
    for (let i = 0; i < bytes.length; i++) counts[bytes[i]]++;
    const expected = bytes.length / 256.0;
    let chi2 = 0;
    for (let i = 0; i < 256; i++) {
        const diff = counts[i] - expected;
        chi2 += (diff * diff) / expected;
    }
    const z = Math.pow(chi2 / 255.0, 1.0 / 3.0) - (1.0 - 2.0 / (9.0 * 255.0));
    const denom = Math.sqrt(2.0 / (9.0 * 255.0));
    const p_value = 0.5 * erfc((z / denom) / Math.SQRT2);
    return { chi2, p_value, pass: p_value >= 0.01 };
}

function calculateEntropy(bytes) {
    const counts = new Map();
    for (const b of bytes) counts.set(b, (counts.get(b) || 0) + 1);
    let ent = 0, total = bytes.length;
    for (const count of counts.values()) {
        const p = count / total;
        ent -= p * Math.log2(p);
    }
    return ent;
}