// data-loader.js
// CSV: label, p0..p783 (0..255), no header
async function readCsvFile(file) {
  const text = await file.text();
  const lines = text.split(/\r?\n/).filter(Boolean);
  const n = lines.length;

  const xs = new Float32Array(n * 784);
  const ys = new Int32Array(n);

  let xi = 0, yi = 0;
  for (const line of lines) {
    const parts = line.split(/[,;\s]+/).filter(Boolean);
    ys[yi++] = parseInt(parts[0], 10);
    for (let j = 1; j <= 784; j++) {
      xs[xi++] = parseFloat(parts[j]) / 255.0;
    }
  }

  const xsTensor = tf.tensor4d(xs, [n, 28, 28, 1]);
  const ysTensor = tf.tensor1d(ys, 'int32');
  return { xs: xsTensor, ys: ysTensor }; // labels unused for denoiser, kept for sampling
}

function addGaussianNoise(x, std = 0.5) {
  // x in [0,1]; return clipped x + N(0, std)
  return tf.tidy(() => {
    const noise = tf.randomNormal(x.shape, 0, std, 'float32');
    const y = tf.add(x, noise);
    return tf.clipByValue(y, 0, 1);
  });
}

function draw28x28ToCanvas(t, canvas, scale = 4) {
  const [h, w] = [28, 28];
  if (!canvas) return;
  canvas.width = w * scale;
  canvas.height = h * scale;
  const ctx = canvas.getContext('2d');
  const data = t.dataSync();
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = Math.round(data[i] * 255);
    img.data[i * 4 + 0] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  // scale up
  const off = document.createElement('canvas');
  off.width = w; off.height = h;
  off.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, w * scale, h * scale);
}

function sampleBatch(xs, k = 5) {
  const n = xs.shape[0];
  const idx = [];
  for (let i = 0; i < k; i++) idx.push(Math.floor(Math.random() * n));
  const gather = tf.tensor1d(idx, 'int32');
  const batch = tf.gather(xs, gather);
  gather.dispose();
  return { batch, idx };
}

export { readCsvFile, addGaussianNoise, draw28x28ToCanvas, sampleBatch };
