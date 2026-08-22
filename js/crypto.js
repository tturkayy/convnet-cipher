const MAGIC = new Uint8Array([0x43, 0x56, 0x50, 0x52]);
const TARGET_MIN_BYTES = 2048;

function zeroize(...buffers) {
    for (const buf of buffers) {
        if (buf && buf.fill) {
            buf.fill(0);
        }
    }
}

async function deriveSubkeys(masterKeyStr, nonce) {
    const enc = new TextEncoder();
    const keyBytes = enc.encode(masterKeyStr);
    let keyMaterial = null;
    let rawBuffer = null;

    try {
        keyMaterial = await crypto.subtle.importKey("raw", keyBytes, { name: "PBKDF2" }, false, ["deriveKey", "deriveBits"]);
        const masterBits = await crypto.subtle.deriveBits({
            name: "PBKDF2", salt: nonce, iterations: 100000, hash: "SHA-256"
        }, keyMaterial, 512);

        rawBuffer = new Uint8Array(masterBits);
        const aesKeyBytes = rawBuffer.slice(0, 32);
        const authKeyBytes = rawBuffer.slice(32, 64);

        const aesKey = await crypto.subtle.importKey("raw", aesKeyBytes, { name: "AES-CTR" }, false, ["encrypt"]);
        const seedStateBuffer = new Uint8Array(rawBuffer);

        zeroize(aesKeyBytes);
        return { aesKey, authKeyBytes, seedStateBuffer };
    } finally {
        zeroize(keyBytes);
        if (rawBuffer) zeroize(rawBuffer);
    }
}

async function generateNeuralKeystream(aesKey, nonce, seedStateBuffer, length) {
    const zeroBuffer = new Uint8Array(length);
    let aesStreamBuffer = null;
    let aesStream = null;

    try {
        aesStreamBuffer = await crypto.subtle.encrypt(
            { name: "AES-CTR", counter: nonce, length: 128 },
            aesKey,
            zeroBuffer
        );
        aesStream = new Uint8Array(aesStreamBuffer);

        const extractor = new JSAdvancedVisionExtractor(0x243F6A88);
        const keystream = new Uint8Array(length);
        let offset = 0, counter = 0;
        const animationFrames = [];

        let currentState = new Int32Array(32 * 32);
        for (let i = 0; i < currentState.length; i++) {
            const pbkdfByte = seedStateBuffer[i % seedStateBuffer.length];
            const aesByte = aesStream[i % aesStream.length];
            currentState[i] = ((pbkdfByte ^ aesByte) & 0xFF) - 128;
        }

        while (offset < length) {
            const res = extractor.forward(currentState);
            
            if (animationFrames.length < 8) {
                animationFrames.push(res.maps);
            }

            const nextState = new Int32Array(32 * 32);
            for (let i = 0; i < nextState.length; i++) {
                const latentVal = res.latent[i % res.latent.length];
                const streamByte = aesStream[(offset + i) % aesStream.length];
                nextState[i] = ((latentVal ^ streamByte) & 0xFF) - 128;
            }

            for (let i = 0; i < res.latent.length && offset < length; i++) {
                const mixed = whitenInt32(res.latent[i] ^ Math.imul(i, 0x517cc1b7) ^ aesStream[offset]);
                keystream[offset] = (aesStream[offset] ^ mixed) & 0xFF;
                offset++;
            }

            currentState = nextState;
            counter++;
        }

        return { keystream, animationFrames };
    } finally {
        zeroize(zeroBuffer);
        if (aesStream) zeroize(aesStream);
    }
}

async function encryptClient(text, keyStr) {
    const raw = new TextEncoder().encode(text);
    const origLen = raw.length;
    let plain = null;
    let nonce = null;
    let authKeyBytes = null;
    let seedStateBuffer = null;
    let keystream = null;

    try {
        if (origLen < TARGET_MIN_BYTES) {
            const padLen = TARGET_MIN_BYTES - origLen;
            const randPad = new Uint8Array(padLen);
            crypto.getRandomValues(randPad);
            plain = new Uint8Array(4 + origLen + padLen);
            new DataView(plain.buffer).setUint32(0, origLen, false);
            plain.set(raw, 4);
            plain.set(randPad, 4 + origLen);
            zeroize(randPad);
        } else {
            plain = new Uint8Array(4 + origLen);
            new DataView(plain.buffer).setUint32(0, origLen, false);
            plain.set(raw, 4);
        }

        nonce = new Uint8Array(16);
        crypto.getRandomValues(nonce);

        const subkeys = await deriveSubkeys(keyStr, nonce);
        authKeyBytes = subkeys.authKeyBytes;
        seedStateBuffer = subkeys.seedStateBuffer;

        const streamRes = await generateNeuralKeystream(subkeys.aesKey, nonce, seedStateBuffer, plain.length);
        keystream = streamRes.keystream;

        const ciphertext = new Uint8Array(plain.length);
        for (let i = 0; i < plain.length; i++) ciphertext[i] = plain[i] ^ keystream[i];

        const payloadBody = new Uint8Array(4 + 16 + ciphertext.length);
        payloadBody.set(MAGIC, 0);
        payloadBody.set(nonce, 4);
        payloadBody.set(ciphertext, 20);

        const authKey = await crypto.subtle.importKey("raw", authKeyBytes, { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
        const tag = new Uint8Array(await crypto.subtle.sign("HMAC", authKey, payloadBody));

        const finalPayload = new Uint8Array(payloadBody.length + 32);
        finalPayload.set(payloadBody, 0);
        finalPayload.set(tag, payloadBody.length);

        return { 
            payload: Array.from(finalPayload).map(b => b.toString(16).padStart(2, '0')).join(''), 
            nonce: new Uint8Array(nonce), 
            animationFrames: streamRes.animationFrames, 
            finalBytes: finalPayload 
        };
    } finally {
        zeroize(raw);
        if (plain) zeroize(plain);
        if (authKeyBytes) zeroize(authKeyBytes);
        if (seedStateBuffer) zeroize(seedStateBuffer);
        if (keystream) zeroize(keystream);
    }
}

async function decryptClient(hexStr, keyStr) {
    const cleanHex = hexStr.replace(/\s+/g, '');
    if (cleanHex.length % 2 !== 0) throw new Error("invalid wire format length");

    const payload = new Uint8Array(cleanHex.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
    if (payload.length < 4 + 16 + 4 + 32 || payload[0] !== 0x43 || payload[1] !== 0x56 || payload[2] !== 0x50 || payload[3] !== 0x52) {
        throw new Error("invalid wire format magic header");
    }

    const nonce = payload.slice(4, 20);
    const tagStart = payload.length - 32;
    const ciphertext = payload.slice(20, tagStart);
    const receivedTag = payload.slice(tagStart);
    const payloadBody = payload.slice(0, tagStart);

    let authKeyBytes = null;
    let seedStateBuffer = null;
    let keystream = null;
    let decryptedBody = null;

    try {
        const subkeys = await deriveSubkeys(keyStr, nonce);
        authKeyBytes = subkeys.authKeyBytes;
        seedStateBuffer = subkeys.seedStateBuffer;

        const authKey = await crypto.subtle.importKey("raw", authKeyBytes, { name: "HMAC", hash: "SHA-256" }, false, ["verify"]);
        const valid = await crypto.subtle.verify("HMAC", authKey, receivedTag, payloadBody);
        if (!valid) throw new Error("mac verification failed: invalid key or corrupted payload");

        const streamRes = await generateNeuralKeystream(subkeys.aesKey, nonce, seedStateBuffer, ciphertext.length);
        keystream = streamRes.keystream;

        decryptedBody = new Uint8Array(ciphertext.length);
        for (let i = 0; i < ciphertext.length; i++) decryptedBody[i] = ciphertext[i] ^ keystream[i];

        const origLen = new DataView(decryptedBody.buffer, decryptedBody.byteOffset, 4).getUint32(0, false);
        return new TextDecoder().decode(decryptedBody.slice(4, 4 + origLen));
    } finally {
        if (authKeyBytes) zeroize(authKeyBytes);
        if (seedStateBuffer) zeroize(seedStateBuffer);
        if (keystream) zeroize(keystream);
        if (decryptedBody) zeroize(decryptedBody);
        zeroize(payload);
    }
}