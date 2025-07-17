const recordIdentify = document.querySelector('.identify');
const recordRegister = document.querySelector('.register');
const outputIdentify = document.querySelector('.output');
const outputRegister = document.querySelector('.reg-output');

let audioContext, processor, source;
let pcmChunks = [];

// Common start/stop logic
async function startCapture() {
    pcmChunks = [];
    audioContext = new AudioContext({ sampleRate: 16000 });
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    source = audioContext.createMediaStreamSource(stream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = e => {
        pcmChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    source.connect(processor);
    processor.connect(audioContext.destination);
}

async function stopCapture() {
    processor.disconnect();
    source.disconnect();
    await audioContext.close();
    // merge chunks
    const total = pcmChunks.reduce((a, c) => a + c.length, 0);
    const fullPCM = new Float32Array(total);
    let offset = 0;
    pcmChunks.forEach(chunk => {
        fullPCM.set(chunk, offset);
        offset += chunk.length;
    });
    return fullPCM.buffer;  // raw ArrayBuffer of Float32 samples
}

// IDENTIFY button
let identifying = false;
recordIdentify.onclick = async () => {
    if (!identifying) {
        await startCapture();
        recordIdentify.textContent = 'Stop & Identify';
        identifying = true;
    } else {
        const buffer = await stopCapture();
        recordIdentify.textContent = 'Record & Identify';
        identifying = false;
        // send to server
        fetch("/get_person", {
            method: "POST",
            headers: { "Content-Type": "application/octet-stream" },
            body: buffer
        })
            .then(r => r.json())
            .then(data => outputIdentify.textContent = data.output)
            .catch(_ => outputIdentify.textContent = "Error");
    }
};

// REGISTER button
let registering = false;
recordRegister.onclick = async () => {
    if (!registering) {
        await startCapture();
        recordRegister.textContent = 'Stop & Register';
        registering = true;
    } else {
        const buffer = await stopCapture();
        recordRegister.textContent = 'Record & Register';
        registering = false;
        const name = document.getElementById('new-name').value.trim();
        if (!name) {
            outputRegister.textContent = 'Enter a name first.';
            return;
        }
        const form = new FormData();
        form.append('name', name);
        form.append('audio', new Blob([buffer], { type: 'application/octet-stream' }), 'audio.bin');

        fetch("/add_person", {
            method: 'POST',
            body: form
        })
            .then(r => r.json())
            .then(data => outputRegister.textContent = data.msg || data.output)
            .catch(_ => outputRegister.textContent = "Error");
    }
};
