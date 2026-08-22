// WebGPU / WebGL Backend Algılama ve Rozet Güncelleme
async function initTensorFlowBackend() {
    const badge = document.getElementById('backendBadge');
    if (typeof tf !== 'undefined') {
        try {
            if (navigator.gpu) {
                await tf.setBackend('webgpu');
                badge.innerText = 'tfjs webgpu active';
                badge.className = 'text-xs font-mono px-2.5 py-1 rounded bg-indigo-950 text-indigo-400 border border-indigo-800/50';
            } else {
                await tf.setBackend('webgl');
                badge.innerText = 'tfjs webgl active';
                badge.className = 'text-xs font-mono px-2.5 py-1 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/50';
            }
            await tf.ready();
        } catch (e) {
            badge.innerText = 'wasm/cpu fallback';
            badge.className = 'text-xs font-mono px-2.5 py-1 rounded bg-slate-900 text-slate-400 border border-slate-800';
        }
    }
}
window.addEventListener('DOMContentLoaded', initTensorFlowBackend);

// Çevrimiçi GPU Eğitimi Olay Yöneticisi
async function handleBrowserTrain() {
    const btn = document.getElementById('btnTrain');
    const statusText = document.getElementById('trainStatusText');
    const pContainer = document.getElementById('trainProgressContainer');
    const pBar = document.getElementById('trainProgressBar');

    btn.disabled = true;
    pContainer.classList.remove('hidden');
    pBar.style.width = '0%';
    statusText.innerText = 'optimizing weights in gpu...';
    statusText.className = 'text-[10px] font-mono text-cyan-400';

    try {
        const t0 = performance.now();
        await trainModelInBrowser((epoch, total, loss) => {
            const pct = Math.floor((epoch / total) * 100);
            pBar.style.width = `${pct}%`;
            statusText.innerText = `epoch ${epoch}/${total} (loss: ${loss.toFixed(6)})`;
        });
        const latency = (performance.now() - t0).toFixed(0);
        statusText.innerText = `✓ fine-tuned in ${latency} ms`;
        statusText.className = 'text-[10px] font-mono text-emerald-400';
    } catch (e) {
        statusText.innerText = 'failed: ' + e.message;
        statusText.className = 'text-[10px] font-mono text-rose-400';
    } finally {
        btn.disabled = false;
        setTimeout(() => pContainer.classList.add('hidden'), 3500);
    }
}

// Renk Haritası
const PALETTES = [
    norm => [Math.floor(norm * 180 + 30), Math.floor(norm * 220 + 35), 255],
    norm => [Math.floor(norm * 68), Math.floor(norm * 200 + 40), Math.floor((1 - norm) * 180 + 75)],
    norm => [Math.floor(norm * 240 + 15), Math.floor(Math.sin(norm * Math.PI) * 160), Math.floor((1 - norm) * 220)],
    norm => [Math.floor(norm * 230), Math.floor(norm * 190), Math.floor((1 - norm) * 150 + 60)],
    norm => [Math.floor(norm * 255), Math.floor(Math.pow(norm, 2) * 180), Math.floor((1 - norm) * 200)],
    norm => [Math.floor(Math.pow(norm, 0.6) * 255), Math.floor(Math.pow(norm, 1.8) * 220), Math.floor(Math.pow(1 - norm, 2) * 120)]
];

let manifoldAnimationTimer = null;

function computeAndRenderLayerMetrics(sampleFrame) {
    if (!sampleFrame || sampleFrame.length !== 6) return;
    sampleFrame.forEach((tensor, idx) => {
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

        const hEl = document.getElementById(`lm_h_${idx}`);
        const sdEl = document.getElementById(`lm_sd_${idx}`);
        const rgEl = document.getElementById(`lm_rg_${idx}`);
        if (hEl) hEl.innerText = `${h.toFixed(3)} b`;
        if (sdEl) sdEl.innerText = stdDev.toFixed(2);
        if (rgEl) rgEl.innerText = `[${min}, ${max}]`;
    });
}

function startManifoldAttractorAnimation(frames) {
    if (!frames || frames.length === 0) return;
    if (manifoldAnimationTimer) clearInterval(manifoldAnimationTimer);

    computeAndRenderLayerMetrics(frames[0]);

    let frameIdx = 0;
    const renderFrame = () => {
        const maps = frames[frameIdx];
        maps.forEach((tensor, idx) => {
            const cvs = document.getElementById(`cvs_${idx}`);
            if (!cvs) return;
            const ctx = cvs.getContext('2d');
            ctx.imageSmoothingEnabled = false;
            
            const w = cvs.width, h = cvs.height;
            const imgData = ctx.createImageData(w, h);
            const colorFunc = PALETTES[idx % PALETTES.length];
            
            let min = Infinity, max = -Infinity;
            for (let i = 0; i < w * h; i++) {
                if (tensor[i] < min) min = tensor[i];
                if (tensor[i] > max) max = tensor[i];
            }
            const range = max === min ? 1 : (max - min);

            for (let i = 0; i < w * h; i++) {
                const norm = (tensor[i] - min) / range;
                const [r, g, b] = colorFunc(norm);
                imgData.data[i * 4] = r;
                imgData.data[i * 4 + 1] = g;
                imgData.data[i * 4 + 2] = b;
                imgData.data[i * 4 + 3] = 255;
            }
            ctx.putImageData(imgData, 0, 0);
        });
        frameIdx = (frameIdx + 1) % frames.length;
    };

    renderFrame();
    if (frames.length > 1) {
        manifoldAnimationTimer = setInterval(renderFrame, 400);
    }
}

function toggleKeyVisibility(inputId, btn) {
    const input = document.getElementById(inputId);
    const isPassword = input.type === 'password';
    input.type = isPassword ? 'text' : 'password';
    
    btn.innerHTML = isPassword ? `
        <svg class="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l18 18" />
        </svg>
    ` : `
        <svg class="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
        </svg>
    `;
}

function runWorkerTask(taskType, payload, key) {
    return new Promise((resolve, reject) => {
        const channel = new MessageChannel();
        channel.port1.onmessage = (e) => {
            if (e.data.success) resolve(e.data.result);
            else reject(new Error(e.data.error));
        };

        setTimeout(async () => {
            try {
                if (taskType === 'encrypt') {
                    const t0 = performance.now();
                    const res = await encryptClient(payload, key);
                    const latency = performance.now() - t0;
                    
                    const entropy = calculateEntropy(res.finalBytes);
                    const mb = runNistMonobit(res.finalBytes);
                    const bf = runNistBlockFrequency(res.finalBytes, 128);
                    const cusum = runNistCusum(res.finalBytes);
                    const rn = runNistRuns(res.finalBytes);
                    const dft = runNistDFT(res.finalBytes);
                    const tmpl = runNistNonOverlappingTemplate(res.finalBytes, 9);
                    const apen = runNistApproximateEntropy(res.finalBytes, 2);
                    const ser = runNistSerial(res.finalBytes, 2);
                    const lin = runNistLinearComplexity(res.finalBytes, 500);
                    const cs = runChiSquare(res.finalBytes);

                    channel.port2.postMessage({
                        success: true,
                        result: {
                            payload: res.payload,
                            nonceHex: '0x' + Array.from(res.nonce.slice(0, 8)).map(b => b.toString(16).padStart(2, '0')).join('') + '...',
                            entropy: entropy.toFixed(4),
                            latency: latency.toFixed(2),
                            animationFrames: res.animationFrames,
                            nist: { mb, bf, cusum, rn, dft, tmpl, apen, ser, lin, cs }
                        }
                    });
                } else if (taskType === 'decrypt') {
                    const plaintext = await decryptClient(payload, key);
                    channel.port2.postMessage({
                        success: true,
                        result: { plaintext }
                    });
                }
            } catch (err) {
                channel.port2.postMessage({ success: false, error: err.message });
            }
        }, 0);
    });
}

async function handleEncrypt() {
    const text = document.getElementById('plainInput').value;
    const key = document.getElementById('encKey').value;
    if (!text || !key) { alert('missing required plaintext or key buffer.'); return; }

    const btn = document.getElementById('btnEncrypt');
    btn.disabled = true;
    btn.innerText = 'computing in worker thread...';

    try {
        const res = await runWorkerTask('encrypt', text, key);

        document.getElementById('cipherOutput').value = res.payload;
        document.getElementById('encOutputBox').classList.remove('hidden');

        document.getElementById('metricEntropy').innerText = `${res.entropy} b/b`;
        document.getElementById('metricNonce').innerText = res.nonceHex;
        document.getElementById('metricLatency').innerText = `${res.latency} ms`;
        document.getElementById('metricsSection').classList.remove('hidden');

        const { mb, bf, cusum, rn, dft, tmpl, apen, ser, lin, cs } = res.nist;

        function updateRow(statId, pvalId, statusId, statStr, pval, pass) {
            document.getElementById(statId).innerText = statStr;
            document.getElementById(pvalId).innerText = `p = ${pval.toFixed(4)}`;
            const el = document.getElementById(statusId);
            el.innerText = pass ? 'passed' : 'failed';
            el.className = pass ? 'py-2 px-3 text-right font-bold text-emerald-400' : 'py-2 px-3 text-right font-bold text-rose-400';
        }

        updateRow('nistStatMonobit', 'nistPvalMonobit', 'nistStatusMonobit', `s_obs = ${mb.s_obs.toFixed(4)}`, mb.p_value, mb.pass);
        updateRow('nistStatBlock', 'nistPvalBlock', 'nistStatusBlock', `χ² = ${bf.chi2.toFixed(2)}`, bf.p_value, bf.pass);
        updateRow('nistStatCusumF', 'nistPvalCusumF', 'nistStatusCusumF', `z = ${cusum.z}`, cusum.p_value, cusum.pass);
        updateRow('nistStatRuns', 'nistPvalRuns', 'nistStatusRuns', `v_n = ${rn.v_n}`, rn.p_value, rn.pass);
        updateRow('nistStatDft', 'nistPvalDft', 'nistStatusDft', `d = ${dft.d.toFixed(4)}`, dft.p_value, dft.pass);
        updateRow('nistStatTemplate', 'nistPvalTemplate', 'nistStatusTemplate', `χ² = ${tmpl.chi2.toFixed(2)}`, tmpl.p_value, tmpl.pass);
        updateRow('nistStatApen', 'nistPvalApen', 'nistStatusApen', `apen = ${apen.apen.toFixed(4)}`, apen.p_value, apen.pass);
        updateRow('nistStatSerial', 'nistPvalSerial', 'nistStatusSerial', `ψ² = ${ser.del1.toFixed(2)}`, ser.p_value, ser.pass);
        updateRow('nistStatLinear', 'nistPvalLinear', 'nistStatusLinear', `χ² = ${lin.chi2.toFixed(2)}`, lin.p_value, lin.pass);
        updateRow('nistStatChi', 'nistPvalChi', 'nistStatusChi', `χ² = ${cs.chi2.toFixed(2)}`, cs.p_value, cs.pass);

        document.getElementById('nistSection').classList.remove('hidden');

        startManifoldAttractorAnimation(res.animationFrames);
        document.getElementById('visSection').classList.remove('hidden');
    } catch (err) {
        alert(err.message);
    } finally {
        btn.disabled = false;
        btn.innerText = 'generate keystream & sign';
    }
}

async function handleDecrypt() {
    const hex = document.getElementById('cipherInput').value;
    const key = document.getElementById('decKey').value;
    if (!hex || !key) { alert('missing required hex payload or key buffer.'); return; }

    const btn = document.getElementById('btnDecrypt');
    btn.disabled = true;
    btn.innerText = 'verifying in worker thread...';

    try {
        const res = await runWorkerTask('decrypt', hex.trim(), key);
        document.getElementById('plainOutput').value = res.plaintext;
        document.getElementById('decOutputBox').classList.remove('hidden');
    } catch (e) {
        alert(e.message);
    } finally {
        btn.disabled = false;
        btn.innerText = 'verify mac & decrypt';
    }
}

function copyToClipboard(id) {
    const el = document.getElementById(id);
    el.select();
    navigator.clipboard.writeText(el.value);
}

// Interactive Particle Background (Dense Cyber Grid)
const canvas = document.getElementById('bg-canvas');
const ctx = canvas.getContext('2d');
let width = canvas.width = window.innerWidth, height = canvas.height = window.innerHeight;

// Yoğunluk ve Bağlantı Parametreleri Artırıldı (45 -> 85 partikül, 120 -> 150px mesafe)
const particles = [];
const numParticles = Math.floor((width * height) / 14000); // Ekrana duyarlı dinamik yoğunluk (~80-100 partikül)
const maxDist = 150;
const mouse = { x: null, y: null, radius: 180 };

window.addEventListener('resize', () => { 
    width = canvas.width = window.innerWidth; 
    height = canvas.height = window.innerHeight; 
});
window.addEventListener('mousemove', (e) => { mouse.x = e.clientX; mouse.y = e.clientY; });
window.addEventListener('mouseleave', () => { mouse.x = null; mouse.y = null; });

class Particle {
    constructor() {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.vx = (Math.random() - 0.5) * 0.8;
        this.vy = (Math.random() - 0.5) * 0.8;
        this.radius = Math.random() * 1.8 + 1.0;
    }
    update() {
        this.x += this.vx;
        this.y += this.vy;
        if (this.x < 0 || this.x > width) this.vx = -this.vx;
        if (this.y < 0 || this.y > height) this.vy = -this.vy;

        if (mouse.x !== null) {
            const dx = mouse.x - this.x;
            const dy = mouse.y - this.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < mouse.radius) {
                const force = (mouse.radius - dist) / mouse.radius;
                this.x -= (dx / dist) * force * 2.0;
                this.y -= (dy / dist) * force * 2.0;
            }
        }
    }
    draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(6, 182, 212, 0.6)';
        ctx.fill();
    }
}

for (let i = 0; i < Math.max(75, numParticles); i++) particles.push(new Particle());

function animate() {
    ctx.clearRect(0, 0, width, height);
    for (let i = 0; i < particles.length; i++) {
        particles[i].update();
        particles[i].draw();
        for (let j = i + 1; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < maxDist) {
                ctx.beginPath();
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(particles[j].x, particles[j].y);
                const alpha = (1 - dist / maxDist) * 0.22;
                ctx.strokeStyle = `rgba(56, 189, 248, ${alpha})`;
                ctx.lineWidth = 1.1;
                ctx.stroke();
            }
        }
    }
    requestAnimationFrame(animate);
}
animate();