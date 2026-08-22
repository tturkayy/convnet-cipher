// Worker Thread İçi Script Yüklemeleri
importScripts('nn-layers.js', 'crypto.js', 'nist.js');

self.onmessage = async (e) => {
    const { action, payload, key, id } = e.data;

    try {
        if (action === 'encrypt') {
            const t0 = performance.now();
            const res = await encryptClient(payload, key);
            const latency = performance.now() - t0;

            const entropy = calculateEntropy(res.finalBytes);
            const mb = runNistMonobit(res.finalBytes);
            const bf = runNistBlockFrequency(res.finalBytes, 128);
            const rn = runNistRuns(res.finalBytes);
            const dft = runNistDFT(res.finalBytes);
            const apen = runNistApproximateEntropy(res.finalBytes, 2);
            const ser = runNistSerial(res.finalBytes, 2);
            const cs = runChiSquare(res.finalBytes);

            // Katman Bazında Entropi ve İstatistiksel Analiz
            const layerMetrics = [];
            if (res.animationFrames && res.animationFrames.length > 0) {
                const sampleFrame = res.animationFrames[0];
                sampleFrame.forEach((tensor) => {
                    let min = Infinity, max = -Infinity, sum = 0, sumSq = 0;
                    const counts = new Map();
                    const len = tensor.length;

                    for (let i = 0; i < len; i++) {
                        const val = tensor[i];
                        if (val < min) min = val;
                        if (val > max) max = val;
                        sum += val;
                        sumSq += val * val;
                        counts.set(val, (counts.get(val) || 0) + 1);
                    }

                    let h = 0;
                    for (const count of counts.values()) {
                        const p = count / len;
                        h -= p * Math.log2(p);
                    }

                    const mean = sum / len;
                    const variance = (sumSq / len) - (mean * mean);
                    const stdDev = Math.sqrt(Math.max(0, variance));

                    layerMetrics.push({
                        entropy: h.toFixed(3),
                        min: min,
                        max: max,
                        stdDev: stdDev.toFixed(2)
                    });
                });
            }

            self.postMessage({
                id,
                success: true,
                action: 'encrypt',
                data: {
                    payload: res.payload,
                    nonceHex: '0x' + Array.from(res.nonce.slice(0, 8)).map(b => b.toString(16).padStart(2, '0')).join('') + '...',
                    entropy: entropy.toFixed(4),
                    latency: latency.toFixed(2),
                    animationFrames: res.animationFrames,
                    layerMetrics,
                    nist: { mb, bf, rn, dft, apen, ser, cs }
                }
            });
        } else if (action === 'decrypt') {
            const plaintext = await decryptClient(payload.trim(), key);
            self.postMessage({
                id,
                success: true,
                action: 'decrypt',
                data: { plaintext }
            });
        }
    } catch (err) {
        self.postMessage({
            id,
            success: false,
            action,
            error: err.message
        });
    }
};