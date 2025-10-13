// app.js
import { readCsvFile, addGaussianNoise, draw28x28ToCanvas, sampleBatch } from './data-loader.js';

let train = { xs: null, ys: null };
let test  = { xs: null, ys: null };
let model = null;

const els = {
  trainCsv: document.getElementById('trainCsv'),
  testCsv: document.getElementById('testCsv'),
  btnLoad: document.getElementById('btnLoad'),
  btnTrain: document.getElementById('btnTrain'),
  btnTest5: document.getElementById('btnTest5'),
  btnEval: document.getElementById('btnEval'),
  noiseStd: document.getElementById('noiseStd'),
  epochs: document.getElementById('epochs'),
  batch: document.getElementById('batch'),
  btnSave: document.getElementById('btnSave'),
  modelJson: document.getElementById('modelJson'),
  modelWeights: document.getElementById('modelWeights'),
  btnLoadModel: document.getElementById('btnLoadModel'),
  btnReset: document.getElementById('btnReset'),
  btnVisor: document.getElementById('btnVisor'),
  dataStatus: document.getElementById('dataStatus'),
  modelInfo: document.getElementById('modelInfo'),
  charts: document.getElementById('charts'),
  logs: document.getElementById('logs'),
  preview: document.getElementById('preview'),
};

function log(msg) {
  const t = new Date().toLocaleTimeString();
  els.logs.textContent += `[${t}] ${msg}\n`;
  els.logs.scrollTop = els.logs.scrollHeight;
}

function setDataStatus() {
  const tN = train.xs ? train.xs.shape[0] : 0;
  const eN = test.xs ? test.xs.shape[0] : 0;
  els.dataStatus.textContent = `train samples: ${tN} • test samples: ${eN}`;
}

function buildAutoencoder() {
  const input = tf.input({ shape: [28, 28, 1] });

  // Encoder
  let x = tf.layers.conv2d({ filters: 32, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu' }).apply(input);
  x = tf.layers.maxPooling2d({ poolSize: 2, strides: 2, padding: 'same' }).apply(x);
  x = tf.layers.conv2d({ filters: 64, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu' }).apply(x);
  x = tf.layers.maxPooling2d({ poolSize: 2, strides: 2, padding: 'same' }).apply(x);

  // Decoder (upsample with transposed conv)
  x = tf.layers.conv2dTranspose({ filters: 64, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' }).apply(x);
  x = tf.layers.conv2dTranspose({ filters: 32, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' }).apply(x);
  const out = tf.layers.conv2d({ filters: 1, kernelSize: 3, padding: 'same', activation: 'sigmoid' }).apply(x);

  const m = tf.model({ inputs: input, outputs: out });
  m.compile({ optimizer: tf.train.adam(), loss: 'meanSquaredError' });

  els.modelInfo.textContent = `params: ${m.countParams().toLocaleString()} • loss: MSE • optimizer: adam`;
  return m;
}

async function trainAutoencoder() {
  if (!train.xs) return log('Load data first.');
  if (!model) model = buildAutoencoder();

  const epochs = parseInt(els.epochs.value, 10);
  const batchSize = parseInt(els.batch.value, 10);
  const std = parseFloat(els.noiseStd.value);

  const [xsTrain, xsVal] = tf.tidy(() => {
    // 90/10 split
    const n = train.xs.shape[0];
    const nTrain = Math.floor(n * 0.9);
    const xTrain = train.xs.slice([0, 0, 0, 0], [nTrain, 28, 28, 1]);
    const xVal = train.xs.slice([nTrain, 0, 0, 0], [n - nTrain, 28, 28, 1]);
    return [xTrain, xVal];
  });

  // Noisy inputs; clean targets = original
  const noisyTrain = addGaussianNoise(xsTrain, std);
  const noisyVal = addGaussianNoise(xsVal, std);

  const surface = { name: 'Loss', tab: 'Charts' };
  const fitCallbacks = tfvis.show.fitCallbacks(surface, ['loss', 'val_loss'], { callbacks: ['onEpochEnd'] });

  log(`Training… epochs=${epochs}, batch=${batchSize}, noise σ=${std}`);
  await model.fit(noisyTrain, xsTrain, {
    epochs, batchSize, shuffle: true, validationData: [noisyVal, xsVal],
    callbacks: fitCallbacks
  });
  log('Training complete.');

  noisyTrain.dispose(); noisyVal.dispose();
  xsTrain.dispose(); xsVal.dispose();
}

function clearPreview() {
  els.preview.innerHTML = '';
}

function makeCol() {
  const col = document.createElement('div');
  col.className = 'strip-col';
  const c1 = document.createElement('canvas'); c1.className = 'preview';
  const c2 = document.createElement('canvas'); c2.className = 'preview';
  const c3 = document.createElement('canvas'); c3.className = 'preview';
  col.appendChild(c1); col.appendChild(c2); col.appendChild(c3);
  return { col, c1, c2, c3 };
}

function renderFive(noisy, recon, clean) {
  clearPreview();
  const k = noisy.shape[0];
  for (let i = 0; i < k; i++) {
    const { col, c1, c2, c3 } = makeCol();
    els.preview.appendChild(col);

    const n = noisy.slice([i,0,0,0],[1,28,28,1]).squeeze();
    const r = recon.slice([i,0,0,0],[1,28,28,1]).squeeze();
    const c = clean.slice([i,0,0,0],[1,28,28,1]).squeeze();

    draw28x28ToCanvas(n, c1, 4);
    draw28x28ToCanvas(r, c2, 4);
    draw28x28ToCanvas(c, c3, 4);

    n.dispose(); r.dispose(); c.dispose();
  }
}

async function testFive() {
  if (!test.xs) return log('Load data first.');
  if (!model) return log('Train or load a model first.');

  const std = parseFloat(els.noiseStd.value);
  const { batch } = sampleBatch(test.xs, 5);
  const noisy = addGaussianNoise(batch, std);
  const recon = model.predict(noisy);

  renderFive(noisy, recon, batch);

  batch.dispose(); noisy.dispose(); recon.dispose();
}

function evaluateMSE() {
  if (!test.xs || !model) return log('Need test data and a model.');
  const std = parseFloat(els.noiseStd.value);

  const pred = tf.tidy(() => {
    const noisy = addGaussianNoise(test.xs, std);
    const y = model.predict(noisy);
    noisy.dispose();
    return y;
  });
  const loss = tf.losses.meanSquaredError(test.xs, pred).mean();
  loss.data().then(v => log(`Test MSE @ σ=${std}: ${v[0].toFixed(6)}`));
  pred.dispose();
}

async function saveModel() {
  if (!model) return log('No model to save.');
  await model.save('downloads://mnist_denoiser');
  log('Model saved (downloaded).');
}

async function loadModelFromFiles() {
  const json = els.modelJson.files[0];
  const bin = els.modelWeights.files[0];
  if (!json || !bin) return log('Choose both model.json and weights.bin.');
  model = await tf.loadLayersModel(tf.io.browserFiles([json, bin]));
  els.modelInfo.textContent = `loaded • params: ${model.countParams().toLocaleString()}`;
  log('Model loaded from files.');
}

function resetAll() {
  clearPreview();
  els.logs.textContent = '';
  els.modelInfo.textContent = '[not built]';
  if (model) { model.dispose(); model = null; }
  if (train.xs) { train.xs.dispose(); train.ys.dispose(); train = { xs:null, ys:null }; }
  if (test.xs)  { test.xs.dispose();  test.ys.dispose();  test  = { xs:null, ys:null }; }
  els.dataStatus.textContent = '[not loaded]';
  tfvis.visor().close();
  log('Reset complete.');
}

function toggleVisor() {
  const v = tfvis.visor();
  v.isOpen() ? v.close() : v.open();
}

async function loadData() {
  if (!els.trainCsv.files[0] || !els.testCsv.files[0]) {
    return log('Pick both Train and Test CSV files.');
  }
  if (train.xs) { train.xs.dispose(); train.ys.dispose(); }
  if (test.xs)  { test.xs.dispose();  test.ys.dispose(); }

  log('Loading data…');
  train = await readCsvFile(els.trainCsv.files[0]);
  test  = await readCsvFile(els.testCsv.files[0]);
  setDataStatus();
  log('Data loaded successfully.');
}

els.btnLoad.addEventListener('click', loadData);
els.btnTrain.addEventListener('click', () => trainAutoencoder().catch(e => log(String(e))));
els.btnTest5.addEventListener('click', () => testFive().catch(e => log(String(e))));
els.btnEval.addEventListener('click', evaluateMSE);
els.btnSave.addEventListener('click', saveModel);
els.btnLoadModel.addEventListener('click', () => loadModelFromFiles().catch(e => log(String(e))));
els.btnReset.addEventListener('click', resetAll);
els.btnVisor.addEventListener('click', toggleVisor);

log('Ready.');
