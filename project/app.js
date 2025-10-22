// ==============================
// Emotion Recognition (TF.js) - COMPLETE app.js
// ==============================

/* Labels you expect from the dataset */
const EMOTIONS = ["joy","sadness","anger","fear","love","surprise"];

/* Global app state */
const state = {
  train: null, val: null, test: null,          // raw items [{text,label}]
  trainT: null, valT: null, testT: null,       // tensors
  tokenizer: null, model: null,
  maxLen: 30, vocabSize: 10000,
};

/* UI elements */
const els = {
  trainFile: document.getElementById('trainFile'),
  valFile: document.getElementById('valFile'),
  testFile: document.getElementById('testFile'),
  btnLoad: document.getElementById('btnLoad'),
  dataStatus: document.getElementById('dataStatus'),

  epochs: document.getElementById('epochs'),
  batch: document.getElementById('batch'),
  vocabSize: document.getElementById('vocabSize'),
  maxLen: document.getElementById('maxLen'),
  btnTrain: document.getElementById('btnTrain'),
  btnEval: document.getElementById('btnEval'),
  btnReset: document.getElementById('btnReset'),

  btnSaveModel: document.getElementById('btnSaveModel'),
  btnSaveTok: document.getElementById('btnSaveTok'),
  modelJson: document.getElementById('modelJson'),
  modelWeights: document.getElementById('modelWeights'),
  tokJson: document.getElementById('tokJson'),
  btnLoadModel: document.getElementById('btnLoadModel'),

  predictText: document.getElementById('predictText'),
  btnPredict: document.getElementById('btnPredict'),

  logs: document.getElementById('logs'),
  results: document.getElementById('results'),
  predOut: document.getElementById('predOut'),
};

/* Logger */
function log(msg){
  els.logs.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
  els.logs.scrollTop = els.logs.scrollHeight;
}

/* --- Tokenization helpers (simple, fast, browser-safe) --- */
function tokenizeBasic(s){
  // lowercase, keep letters/numbers/underscore + spaces, drop punctuation
  return s.toLowerCase().replace(/[^\w\s’']/g,' ').split(/\s+/).filter(Boolean);
}

/* --- Robust dataset reader for .txt (tab or other delimiters) --- */
async function readTxtFile(file){
  const txt = await file.text();
  const lines = txt
    .split(/\r?\n/)
    .map(l => l.replace(/^\uFEFF/, '')) // strip BOM
    .filter(l => l.trim().length);

  const items = [];
  for (let i=0;i<lines.length;i++){
    const raw = lines[i];

    // Try tab, comma, semicolon, or 2+ spaces
    let parts = raw.split(/\t|,|;|\s{2,}/);
    if (parts.length < 2){
      // fallback: split on last whitespace chunk
      const m = raw.match(/^(.*?)[\t ]+([^\t ]+)$/);
      if (m) parts = [m[1], m[2]];
    }
    if (parts.length < 2) continue;

    const first = parts[0].trim().toLowerCase();
    const last  = parts[parts.length-1].trim().toLowerCase();

    // Skip header like "text<TAB>label"
    if (i === 0 && (first === 'text' || first === 'sentence') && last === 'label') continue;

    // Assume label is last field
    const label = last;
    const text  = parts.slice(0, parts.length-1).join(' ').trim();

    if (!text || !EMOTIONS.includes(label)) continue; // filter unknown labels
    items.push({ text, label });
  }
  return items;
}

/* --- Tokenizer (frequency based) --- */
function buildTokenizer(samples, vocabSize){
  const freq = new Map();
  for (const s of samples){
    for (const t of tokenizeBasic(s)){
      freq.set(t, (freq.get(t)||0) + 1);
    }
  }
  const sorted = [...freq.entries()].sort((a,b)=>b[1]-a[1]).slice(0, vocabSize-2);
  const wordIndex = Object.create(null); // 0=PAD, 1=OOV
  let idx = 2;
  for (const [w] of sorted){ wordIndex[w] = idx++; }

  return {
    wordIndex,
    toSeq(text, maxLen){
      const toks = tokenizeBasic(text);
      const arr = new Array(maxLen).fill(0);
      for (let i=0;i<Math.min(toks.length,maxLen);i++){
        arr[i] = wordIndex[toks[i]] || 1; // OOV -> 1
      }
      return arr;
    }
  };
}

/* --- Convert items -> tensors --- */
function toXY(items, tokenizer, maxLen){
  const X = new Float32Array(items.length * maxLen);
  const y = new Int32Array(items.length);
  for (let i=0;i<items.length;i++){
    const seq = tokenizer.toSeq(items[i].text, maxLen);
    X.set(seq, i*maxLen);
    y[i] = EMOTIONS.indexOf(items[i].label);
  }
  // ⬇ change dtype to float32 here
  const xs = tf.tensor2d(X, [items.length, maxLen], 'float32');
  const ys = tf.tensor1d(y, 'int32');
  return { xs, ys };
}


/* --- CNN + MLP Hybrid model --- */
function buildModel(vocabSize, maxLen, numClasses){
  const input = tf.input({shape:[maxLen]});

  // Embedding
  let x = tf.layers.embedding({
    inputDim: vocabSize,
    outputDim: 128,
    inputLength: maxLen
  }).apply(input); // [B, L, 128]

  // Multi-kernel Conv1D + GlobalMaxPool
  const convs = [3,4,5].map(k => {
    const c = tf.layers.conv1d({
      filters: 128, kernelSize: k, activation: 'relu', padding: 'valid'
    }).apply(x);
    return tf.layers.globalMaxPooling1d().apply(c); // [B, 128]
  });

  // Concatenate + Dropout
  let feat = tf.layers.concatenate().apply(convs);  // [B, 384]
  feat = tf.layers.dropout({rate: 0.5}).apply(feat);

  // MLP classifier head
  feat = tf.layers.dense({units:128, activation:'relu'}).apply(feat);
  feat = tf.layers.dropout({rate:0.3}).apply(feat);
  feat = tf.layers.dense({units:64, activation:'relu'}).apply(feat);

  const out = tf.layers.dense({units:numClasses, activation:'softmax'}).apply(feat);

  const model = tf.model({inputs: input, outputs: out});
  model.compile({
    optimizer: tf.train.adam(1e-3),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });
  return model;
}

/* --- Confusion matrix renderer --- */
function showConfusionMatrix(yTrue, yPred){
  const K = EMOTIONS.length;
  const M = Array.from({length:K}, ()=>Array(K).fill(0));
  for (let i=0;i<yTrue.length;i++) M[yTrue[i]][yPred[i]]++;
  let html = `<table class="cm"><tr><th></th>${EMOTIONS.map(e=>`<th>${e}</th>`).join('')}</tr>`;
  for (let r=0;r<K;r++){ html += `<tr><th>${EMOTIONS[r]}</th>${M[r].map(n=>`<td>${n}</td>`).join('')}</tr>`; }
  html += `</table>`;
  els.results.innerHTML = html;
}

/* --- Prediction bars --- */
function renderPrediction(probs){
  const arr = Array.from(probs);
  const pairs = EMOTIONS.map((e,i)=>({emo:e,p:arr[i]})).sort((a,b)=>b.p-a.p);
  let html = `<div class="bars">`;
  for (const {emo,p} of pairs){
    html += `<div>${emo}</div><div class="bar"><div class="fill" style="width:${(p*100).toFixed(1)}%"></div></div>`;
  }
  html += `</div><div class="top">Top: <b>${pairs[0].emo}</b> (${(pairs[0].p*100).toFixed(1)}%)</div>`;
  els.predOut.innerHTML = html;
}

/* ================== BUTTON HANDLERS ================== */

els.btnLoad.onclick = async () => {
  try{
    if (!els.trainFile.files[0]){ log('Select train.txt'); return; }

    state.train = await readTxtFile(els.trainFile.files[0]);
    state.val   = els.valFile.files[0]  ? await readTxtFile(els.valFile.files[0])  : null;
    state.test  = els.testFile.files[0] ? await readTxtFile(els.testFile.files[0]) : null;

    state.vocabSize = parseInt(els.vocabSize.value,10) || 10000;
    state.maxLen    = parseInt(els.maxLen.value,10) || 30;

    // If no rows parsed, explain & stop
    if (state.train.length === 0){
      els.dataStatus.textContent = `Loaded: train=0, val=${state.val?state.val.length:0}, test=${state.test?state.test.length:0}`;
      log('ERROR: 0 training rows parsed. Ensure each line is "text<TAB>label" and labels are one of: ' + EMOTIONS.join(', '));
      return;
    }

    state.tokenizer = buildTokenizer(state.train.map(d=>d.text), state.vocabSize);

    state.trainT = toXY(state.train, state.tokenizer, state.maxLen);
    state.valT   = state.val  ? toXY(state.val,  state.tokenizer, state.maxLen) : null;
    state.testT  = state.test ? toXY(state.test, state.tokenizer, state.maxLen) : null;

    els.dataStatus.textContent =
      `Loaded: train=${state.train.length}` +
      (state.val?`, val=${state.val.length}`:'') +
      (state.test?`, test=${state.test.length}`:'') +
      ` • vocab=${state.vocabSize} • maxLen=${state.maxLen}`;

    log(`Data loaded and tokenized. First: "${state.train[0].text}" → ${state.train[0].label}`);
  }catch(err){ log('ERROR loading: '+err.message); console.error(err); }
};

els.btnTrain.onclick = async () => {
  try{
    if (!state.trainT || state.train.length === 0){
      return log('Cannot train: 0 training samples. Fix dataset parsing first.');
    }
    if (state.model){ state.model.dispose(); state.model = null; }

    const epochs = parseInt(els.epochs.value,10) || 10;
    const batch  = parseInt(els.batch.value,10)  || 64;

    state.model = buildModel(state.vocabSize, state.maxLen, EMOTIONS.length);

    // Use 'accuracy' keys (newer TFJS)
    const callbacks = tfvis.show.fitCallbacks(
      { name: 'Training', tab: 'Charts' },
      ['loss','val_loss','accuracy','val_accuracy'],
      { callbacks: ['onEpochEnd'] }
    );

    log(`Training… epochs=${epochs}, batch=${batch}`);
    await state.model.fit(state.trainT.xs, state.trainT.ys, {
      epochs, batchSize: batch,
      validationData: state.valT ? [state.valT.xs, state.valT.ys] : null,
      shuffle: true,
      callbacks
    });
    log('Training complete.');
  }catch(err){ log('ERROR training: '+err.message); console.error(err); }
};

els.btnEval.onclick = async () => {
  try{
    if (!state.model) return log('Train or load a model first.');
    const set = state.testT || state.valT || state.trainT;
    if (!set) return log('No dataset tensors available.');

    const probs = state.model.predict(set.xs);
    const yhat  = probs.argMax(-1);
    const ypred = Array.from(await yhat.data());
    const ytrue = Array.from(await set.ys.data());
    const acc = ypred.filter((p,i)=>p===ytrue[i]).length / ytrue.length;

    log(`Eval accuracy: ${(acc*100).toFixed(2)}% on ${ytrue.length} samples.`);
    showConfusionMatrix(ytrue, ypred);

    probs.dispose(); yhat.dispose();
  }catch(err){ log('ERROR eval: '+err.message); console.error(err); }
};

els.btnPredict.onclick = () => {
  try{
    const txt = els.predictText.value.trim();
    if (!txt) return;
    if (!state.model || !state.tokenizer) return log('Load/train model first.');

    const seq = state.tokenizer.toSeq(txt, state.maxLen);
    const xs  = tf.tensor2d([seq], [1,state.maxLen], 'int32');
    const p   = state.model.predict(xs);
    p.data().then(arr => renderPrediction(arr));
    p.dispose(); xs.dispose();
  }catch(err){ log('ERROR predict: '+err.message); console.error(err); }
};

els.btnSaveModel.onclick = async () => {
  if (!state.model) return log('No model to save.');
  await state.model.save('downloads://emotion_mlp');
  log('Model saved (downloaded).');
};

els.btnSaveTok.onclick = () => {
  if (!state.tokenizer) return log('No tokenizer to save.');
  const blob = new Blob([JSON.stringify({
    wordIndex: state.tokenizer.wordIndex,
    maxLen: state.maxLen, vocabSize: state.vocabSize
  }, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'tokenizer.json';
  a.click();
  URL.revokeObjectURL(a.href);
  log('Tokenizer saved.');
};

els.btnLoadModel.onclick = async () => {
  try{
    const jf = els.modelJson.files[0], wf = els.modelWeights.files[0], tfj = els.tokJson.files[0];
    if (!jf || !wf || !tfj) return log('Pick model.json, weights.bin, and tokenizer.json');

    // load model
    if (state.model) state.model.dispose();
    state.model = await tf.loadLayersModel(tf.io.browserFiles([jf,wf]));

    // load tokenizer
    const tok = JSON.parse(await tfj.text());
    state.maxLen = tok.maxLen || state.maxLen;
    state.vocabSize = tok.vocabSize || state.vocabSize;
    state.tokenizer = {
      wordIndex: tok.wordIndex || {},
      toSeq(text, maxLen=state.maxLen){
        const toks = tokenizeBasic(text);
        const arr = new Array(maxLen).fill(0);
        for (let i=0;i<Math.min(toks.length,maxLen);i++){
          const id = this.wordIndex[toks[i]] || 1;
          arr[i] = id;
        }
        return arr;
      }
    };
    log('Model + tokenizer loaded.');
  }catch(err){ log('ERROR loading model: '+err.message); console.error(err); }
};

els.btnReset.onclick = () => {
  try{
    if (state.model){ state.model.dispose(); state.model = null; }
    ['train','val','test','trainT','valT','testT'].forEach(k => state[k]=null);
    state.tokenizer = null;
    els.logs.textContent = '';
    els.results.innerHTML = '';
    els.predOut.innerHTML = '';
    els.dataStatus.textContent = 'No data loaded';
    tfvis.visor().close();
    log('Reset done.');
  }catch(e){ log('ERROR reset: '+e.message); }
};
