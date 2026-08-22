// Aktivasyon & Dönüşüm Yardımcıları
const leakyRelu = (data) => {
    const out = new Int32Array(data.length);
    for (let i = 0; i < data.length; i++) {
        const v = data[i];
        out[i] = v > 0 ? v : (Math.imul(v, 26) >> 8);
    }
    return out;
};

function fixedTanh(val) {
    const absVal = Math.abs(val);
    const soft = Math.floor((Math.imul(val, 128)) / (absVal + 128));
    return Math.max(-128, Math.min(127, soft));
}

function fixedSigmoid(val) {
    if (val < -1024) return 0;
    if (val > 1024) return 256;
    return Math.floor((val + 1024) / 8);
}

function whitenInt32(x) {
    x = (x ^ (x >>> 16)) >>> 0;
    x = Math.imul(x, 0x85ebca6b) >>> 0;
    x = (x ^ (x >>> 13)) >>> 0;
    x = Math.imul(x, 0xc2b2ae35) >>> 0;
    x = (x ^ (x >>> 16)) >>> 0;
    return x >>> 0;
}

function createNumsWeightGenerator(seed = 0x243F6A88) {
    let state = seed >>> 0;
    return {
        nextInt(min, max) {
            state = (Math.imul(1103515245, state) + 12345) >>> 0;
            const norm = (state >>> 16) / 65536.0;
            return Math.floor(min + norm * (max - min));
        }
    };
}

let TRAINED_WEIGHTS = typeof EMBEDDED_TRAINED_WEIGHTS !== 'undefined' ? EMBEDDED_TRAINED_WEIGHTS : null;

// TensorFlow.js WebGPU / WebGL Hızlandırılmış Evrişim Katmanı
class JSConv2D {
    constructor(inC, outC, kSize, stride, rng, customWeights = null, customBias = null) {
        this.inC = inC; this.outC = outC; this.kSize = kSize; this.stride = stride;
        this.pad = stride === 1 ? (kSize >> 1) : 0;
        this.weights = customWeights ? new Int32Array(customWeights) : new Int32Array(outC * inC * kSize * kSize);
        this.bias = customBias ? new Int32Array(customBias) : new Int32Array(outC);

        if (!customWeights) {
            for (let i = 0; i < this.weights.length; i++) this.weights[i] = rng.nextInt(-128, 128);
        }
        if (!customBias) {
            for (let i = 0; i < this.bias.length; i++) this.bias[i] = rng.nextInt(-64, 64);
        }

        this.tfFilter = null;
        this.tfBias = null;
        if (typeof tf !== 'undefined') {
            try {
                const floatWeights = new Float32Array(kSize * kSize * inC * outC);
                for (let oc = 0; oc < outC; oc++) {
                    for (let ic = 0; ic < inC; ic++) {
                        for (let kh = 0; kh < kSize; kh++) {
                            for (let kw = 0; kw < kSize; kw++) {
                                const srcIdx = oc * (inC * kSize * kSize) + ic * (kSize * kSize) + kh * kSize + kw;
                                const dstIdx = kh * (kSize * inC * outC) + kw * (inC * outC) + ic * outC + oc;
                                floatWeights[dstIdx] = this.weights[srcIdx];
                            }
                        }
                    }
                }
                const floatBias = new Float32Array(outC);
                for (let oc = 0; oc < outC; oc++) floatBias[oc] = this.bias[oc];

                this.tfFilter = tf.tensor4d(floatWeights, [kSize, kSize, inC, outC]);
                this.tfBias = tf.tensor1d(floatBias);
            } catch (e) {
                this.tfFilter = null;
            }
        }
    }

    forward(x, h, w) {
        if (typeof tf !== 'undefined' && this.tfFilter) {
            return tf.tidy(() => {
                const floatX = new Float32Array(h * w * this.inC);
                for (let ic = 0; ic < this.inC; ic++) {
                    for (let i = 0; i < h * w; i++) {
                        floatX[i * this.inC + ic] = x[ic * h * w + i];
                    }
                }
                const inTensor = tf.tensor4d(floatX, [1, h, w, this.inC]);
                const padding = this.stride === 1 ? 'same' : 'valid';
                const conv = tf.conv2d(inTensor, this.tfFilter, this.stride, padding);
                const biased = tf.add(conv, this.tfBias);
                const [_, hOut, wOut, cOut] = biased.shape;
                const rawArr = biased.dataSync();

                const out = new Int32Array(cOut * hOut * wOut);
                for (let oc = 0; oc < cOut; oc++) {
                    for (let i = 0; i < hOut * wOut; i++) {
                        out[oc * hOut * wOut + i] = Math.floor(rawArr[i * cOut + oc]) >> 8;
                    }
                }
                return { data: out, h: hOut, w: wOut, c: cOut };
            });
        }

        const pad = this.pad;
        const hOut = Math.floor((h + 2 * pad - this.kSize) / this.stride) + 1;
        const wOut = Math.floor((w + 2 * pad - this.kSize) / this.stride) + 1;
        const out = new Int32Array(this.outC * hOut * wOut);

        for (let oc = 0; oc < this.outC; oc++) {
            for (let oh = 0; oh < hOut; oh++) {
                for (let ow = 0; ow < wOut; ow++) {
                    let acc = this.bias[oc];
                    for (let ic = 0; ic < this.inC; ic++) {
                        for (let kh = 0; kh < this.kSize; kh++) {
                            for (let kw = 0; kw < this.kSize; kw++) {
                                const ih = oh * this.stride + kh - pad;
                                const iw = ow * this.stride + kw - pad;
                                if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                    const inVal = x[ic * h * w + ih * w + iw];
                                    const wVal = this.weights[oc * (this.inC * this.kSize * this.kSize) + ic * (this.kSize * this.kSize) + kh * this.kSize + kw];
                                    acc += Math.imul(inVal, wVal);
                                }
                            }
                        }
                    }
                    out[oc * hOut * wOut + oh * wOut + ow] = (acc >> 8);
                }
            }
        }
        return { data: out, h: hOut, w: wOut, c: this.outC };
    }
}

class JSInceptionBlock {
    constructor(inC, outBranchC, rng, wDict = null) {
        this.inC = inC;
        this.branch1x1 = new JSConv2D(inC, outBranchC, 1, 1, rng, wDict?.inc_b1_w, wDict?.inc_b1_b);
        this.branch3x3 = new JSConv2D(inC, outBranchC, 3, 1, rng, wDict?.inc_b2_w, wDict?.inc_b2_b);
        this.branch5x5 = new JSConv2D(inC, outBranchC, 5, 1, rng, wDict?.inc_b3_w, wDict?.inc_b3_b);
        this.outC = outBranchC * 3;
        this.projShortcut = new JSConv2D(inC, this.outC, 1, 1, rng, wDict?.inc_proj_w, wDict?.inc_proj_b);
    }
    forward(x, h, w) {
        const b1 = this.branch1x1.forward(x, h, w);
        const b2 = this.branch3x3.forward(x, h, w);
        const b3 = this.branch5x5.forward(x, h, w);
        const out = new Int32Array(this.outC * h * w);

        const size = h * w;
        out.set(b1.data.subarray(0, b1.c * size), 0);
        out.set(b2.data.subarray(0, b2.c * size), b1.c * size);
        out.set(b3.data.subarray(0, b3.c * size), (b1.c + b2.c) * size);

        const shortcut = this.projShortcut.forward(x, h, w);
        for (let i = 0; i < out.length; i++) {
            out[i] = (out[i] + shortcut.data[i]) >> 1;
        }

        return { data: leakyRelu(out), h, w, c: this.outC };
    }
}

class JSEModule {
    constructor(channels, reduction, rng, wDict = null) {
        this.channels = channels;
        this.reduced = Math.max(1, Math.floor(channels / reduction));
        this.fc1 = wDict?.se_fc1_w ? new Int32Array(wDict.se_fc1_w) : new Int32Array(this.channels * this.reduced);
        this.fc2 = wDict?.se_fc2_w ? new Int32Array(wDict.se_fc2_w) : new Int32Array(this.reduced * this.channels);
        
        if (!wDict) {
            for (let i = 0; i < this.fc1.length; i++) this.fc1[i] = rng.nextInt(-64, 64);
            for (let i = 0; i < this.fc2.length; i++) this.fc2[i] = rng.nextInt(-64, 64);
        }
    }
    forward(x, h, w) {
        const size = h * w;
        const out = new Int32Array(x.length);

        const gap = new Int32Array(this.channels);
        for (let c = 0; c < this.channels; c++) {
            let sum = 0;
            for (let i = 0; i < size; i++) sum += x[c * size + i];
            gap[c] = Math.floor(sum / size);
        }

        const hidden = new Int32Array(this.reduced);
        for (let r = 0; r < this.reduced; r++) {
            let acc = 0;
            for (let c = 0; c < this.channels; c++) acc += Math.imul(gap[c], this.fc1[r * this.channels + c]);
            hidden[r] = Math.max(0, acc >> 8);
        }

        const weights = new Int32Array(this.channels);
        for (let c = 0; c < this.channels; c++) {
            let acc = 0;
            for (let r = 0; r < this.reduced; r++) acc += Math.imul(hidden[r], this.fc2[c * this.reduced + r]);
            weights[c] = fixedSigmoid(acc >> 6);
        }

        for (let c = 0; c < this.channels; c++) {
            const wScale = weights[c];
            for (let i = 0; i < size; i++) {
                const idx = c * size + i;
                const scaled = (Math.imul(x[idx], wScale)) >> 8;
                out[idx] = (scaled + x[idx]) >> 1;
            }
        }
        return { data: out, h, w, c: this.channels };
    }
}

class JSDepthwiseSeparableConv {
    constructor(inC, outC, stride, rng, dwW, dwB, pwW, pwB, projW, projB) {
        this.inC = inC; this.outC = outC; this.stride = stride;
        this.dwConv = new JSConv2D(inC, inC, 3, stride, rng, dwW, dwB);
        this.pwConv = new JSConv2D(inC, outC, 1, 1, rng, pwW, pwB);
        if (inC !== outC || stride !== 1) {
            this.projShortcut = new JSConv2D(inC, outC, 1, stride, rng, projW, projB);
        } else {
            this.projShortcut = null;
        }
    }
    forward(x, h, w) {
        const dw = this.dwConv.forward(x, h, w);
        const dwAct = leakyRelu(dw.data);
        const pw = this.pwConv.forward(dwAct, dw.h, dw.w);
        const out = pw.data;

        if (this.projShortcut) {
            const shortcut = this.projShortcut.forward(x, h, w);
            for (let i = 0; i < out.length; i++) {
                out[i] = (out[i] + shortcut.data[i]) >> 1;
            }
        } else {
            for (let i = 0; i < out.length; i++) {
                out[i] = (out[i] + x[i]) >> 1;
            }
        }

        return { data: leakyRelu(out), h: pw.h, w: pw.w, c: this.outC };
    }
}

class JSAdvancedVisionExtractor {
    constructor(fixedSeed = 0x243F6A88) {
        const rng = createNumsWeightGenerator(fixedSeed);
        const w = TRAINED_WEIGHTS;
        this.inception = new JSInceptionBlock(1, 3, rng, w);
        this.seBlock = new JSEModule(9, 3, rng, w);
        this.dwStage1 = new JSDepthwiseSeparableConv(9, 16, 2, rng, w?.dw1_dw_w, w?.dw1_dw_b, w?.dw1_pw_w, w?.dw1_pw_b, w?.dw1_proj_w, w?.dw1_proj_b);
        this.dwStage2 = new JSDepthwiseSeparableConv(16, 16, 1, rng, w?.dw2_dw_w, w?.dw2_dw_b, w?.dw2_pw_w, w?.dw2_pw_b, null, null);
        this.latentBottleneck = new JSConv2D(16, 4, 1, 1, rng, w?.btn_w, w?.btn_b);
    }
    forward(inputState) {
        const inc = this.inception.forward(inputState, 32, 32);
        const se = this.seBlock.forward(inc.data, inc.h, inc.w);
        const dw1 = this.dwStage1.forward(se.data, se.h, se.w);
        const dw2 = this.dwStage2.forward(dw1.data, dw1.h, dw1.w);
        const latentRaw = this.latentBottleneck.forward(dw2.data, dw2.h, dw2.w);

        const latentTanh = new Int32Array(latentRaw.data.length);
        for (let i = 0; i < latentRaw.data.length; i++) {
            latentTanh[i] = fixedTanh(latentRaw.data[i]);
        }

        return {
            maps: [inputState, inc.data, se.data, dw1.data, dw2.data, latentTanh],
            latent: latentTanh
        };
    }
}

// Tarayıcı İçi Çevrimiçi İnce Ayar Motoru (In-Browser Online Fine-Tuning)
async function trainModelInBrowser(onEpochProgress) {
    if (typeof tf === 'undefined') {
        throw new Error("TensorFlow.js engine not available in browser.");
    }

    const steps = 100;
    const optimizer = tf.train.adam(0.002);

    // Kriptografik Düzgünlük & Lag-1 Korelasyon Kaybı
    function cryptanalyticLoss(pred) {
        return tf.tidy(() => {
            const flat = tf.reshape(pred, [-1]);
            const normY = tf.mul(tf.add(flat, 1.0), 0.5);

            const mean = tf.mean(normY);
            const lossMean = tf.square(tf.sub(mean, 0.5));
            const variance = tf.mean(tf.square(tf.sub(normY, mean)));
            const lossVar = tf.square(tf.sub(variance, 1.0 / 12.0));

            const y1 = normY.slice([0], [normY.shape[0] - 1]);
            const y2 = normY.slice([1], [normY.shape[0] - 1]);
            const autocorr = tf.div(
                tf.sum(tf.mul(tf.sub(y1, mean), tf.sub(y2, mean))),
                tf.add(tf.sum(tf.square(tf.sub(normY, mean))), 1e-8)
            );
            const lossAuto = tf.square(autocorr);

            return tf.add(tf.add(lossMean, lossVar), tf.mul(lossAuto, 0.5));
        });
    }

    // Doğru TF.js Şekli: [kH, kW, inC, outC] -> [1, 1, 16, 4]
    const trainableWeight = tf.variable(tf.randomNormal([1, 1, 16, 4], 0, 0.5));

    for (let epoch = 1; epoch <= steps; epoch++) {
        let currentLoss = 0;
        await tf.tidy(() => {
            optimizer.minimize(() => {
                const dummyBatch = tf.randomUniform([4, 16, 16, 16], -1.0, 1.0);
                const conv = tf.conv2d(dummyBatch, trainableWeight, 1, 'same');
                const out = tf.tanh(conv);
                const loss = cryptanalyticLoss(out);
                currentLoss = loss.dataSync()[0];
                return loss;
            });
        });

        if (onEpochProgress && epoch % 10 === 0) {
            onEpochProgress(epoch, steps, currentLoss);
            await tf.nextFrame();
        }
    }

    // Güncellenmiş ağırlıkları yerel önbelleğe aktar
    const updated = trainableWeight.dataSync();
    if (!TRAINED_WEIGHTS) TRAINED_WEIGHTS = {};
    TRAINED_WEIGHTS.btn_w = Array.from(updated).map(v => Math.max(-128, Math.min(127, Math.round(v * 127))));
    trainableWeight.dispose();
}