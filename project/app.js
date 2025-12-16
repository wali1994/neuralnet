// app.js — Emotion Recognition (TF.js, MLP-only, stable)

// ---------- Elements ----------
const els = {
  trainFile: document.getElementById('trainFile'),
  valFile: document.getElementById('valFile'),
  testFile: document.getElementById('testFile'),

  btnLoad: document.getElementById('btnLoad'),
  btnTrain: document.getElementById('btnTrain'),
  btnEval: document.getElementById('btnEval'),
  btnReset: document.getElementById('btnReset'),

  dataStatus: document.getElementById('dataStatus'),
  logs: document.getElementById('logs'),

  vocabSize: document.getElementById('vocabSize'),
  maxLen: document.getElementById('maxLen'),
  epochs: document.getElementById('epochs'),
  batch: document.getElementById('batch'),

  predictText: document.getElementById('predictText'),
  btnPredict: document.getElementById('btnPredict'),
  predOut: document.getElementById('predOut'),

  // optional model i/o (ignore if not on page)
  btnSaveModel: document.getElementById('btnSaveModel'),
  btnSaveTok: document.getElementById('btnSaveTok'),
  modelJson: document.getElementById('modelJson'),
  modelWeights: document.getElementById('modelWeights'),
  tokJson: document.getElementById('tokJson'),
  btnLoadModel: document.getElementById('btnLoadModel'),

  results: document.getElementById('results'),
};

// ---------- State ----------
let LABELS = [];                  // inferred from train file
let label2id = Object.create(null);

const state = {
  train: null, val: null, test: null,     // raw [{text,label}]
  trainT: null, valT: null, testT: null,  // tensors {xs,ys}
  tokenizer: null, model: null,
  maxLen: 30, vocabSize: 10000,
};

// ---------- Utils ----------
function log(msg){
  if (!els.logs) return console.log(msg);
  els.logs.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
  els.logs.scrollTop = els.logs.scrollHeight;
}

function tokenizeBasic(s){
  return s.toLowerCase().replace(/[^\w\s’']/g, ' ').split(/\s+/).filter(Boolean);
}

// Read semicolon dataset: "text;label"
async function readTxtFile(file){
  const t = await file.text();
  return t.split(/\r?\n/)
          .map(l => l.trim())
          .filter(l => l.length)
          .map(l => {
            const idx = l.lastIndexOf(';');
            if (idx === -1) return null;
            const text = l.slice(0, idx).trim().toLowerCase();
            const label = l.slice(idx+1).trim().toLowerCase();
            if (!text || !label) return null;
            return { text, label };
          })
          .filter(Boolean);
}

function inferLabelsFromTrain(trainItems){
  const set = new Set(trainItems.map(d => d.label));
  LABELS = Array.from(set).sort();  // stable order
  label2id = Object.create(null);
  LABELS.forEach((lab,i)=> label2id[lab] = i);
  log(`Labels: [${LABELS.join(', ')}]`);
  return LABELS.length;
}

function buildTokenizer(texts, vocabSize){
  const freq = new Map();
  for (const s of texts){
    for (const t of tokenizeBasic(s)) freq.set(t, (freq.get(t) || 0) + 1);
  }
  const sorted = [...freq.entries()].sort((a,b)=>b[1]-a[1]).slice(0, vocabSize-2);
  const wordIndex = Object.create(null); // 0=PAD, 1=OOV
  let idx = 2;
  for (const [w] of sorted) wordIndex[w] = idx++;
  return {
    wordIndex,
    toSeq(text, maxLen){
      const toks = tokenizeBasic(text);
      const out = new Array(maxLen).fill(0);
      for (let i=0;i<Math.min(toks.length,maxLen);i++){
        out[i] = wordIndex[toks[i]] || 1;
      }
      return out;
    }
  };
}

function toXY(items, tokenizer, maxLen){
  const X = new Int32Array(items.length * maxLen);
  const y = new Int32Array(items.length);
  for (let i=0;i<items.length;i++){
    const seq = tokenizer.toSeq(items[i].text, maxLen);
    X.set(seq, i*maxLen);
    y[i] = label2id[items[i].label];
  }
  const xs = tf.tensor2d(X, [items.length, maxLen], 'int32'); // embedding likes int32
  const ys = tf.tensor1d(y, 'int32');
  return { xs, ys };
}

// ---------- Model (MLP only; no conv, no dropout) ----------
function buildModel(vocabSize, maxLen, numClasses){
  const input = tf.input({ shape: [maxLen], dtype: 'int32' });

  // Embedding -> float32 automatically
  let x = tf.layers.embedding({
    inputDim: vocabSize,
    outputDim: 128,
    inputLength: maxLen
  }).apply(input);

  // Global average pool over time
  x = tf.layers.globalAveragePooling1d().apply(x);

  // Dense layers (no dropout)
  x = tf.layers.dense({ units: 128, activation: 'relu' }).apply(x);
  x = tf.layers.dense({ units: 64, activation: 'relu' }).apply(x);

  const out = tf.layers.dense({ units: numClasses, activation: 'softmax' }).apply(x);

  const model = tf.model({ inputs: input, outputs: out });
  model.compile({
    optimizer: tf.train.adam(1e-3),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });
  return model;
}

// ---------- UI helpers ----------
function showConfusionMatrix(yTrue, yPred){
  if (!els.results) return;
  const K = LABELS.length;
  const M = Array.from({length:K}, ()=>Array(K).fill(0));
  for (let i=0;i<yTrue.length;i++) M[yTrue[i]][yPred[i]]++;
  let html = `<table class="cm"><tr><th></th>${LABELS.map(e=>`<th>${e}</th>`).join('')}</tr>`;
  for (let r=0;r<K;r++){
    html += `<tr><th>${LABELS[r]}</th>${M[r].map(n=>`<td>${n}</td>`).join('')}</tr>`;
  }
  html += `</table>`;
  els.results.innerHTML = html;
}

function renderPrediction(probArray){
  if (!els.predOut) return;
  const pairs = LABELS.map((e,i)=>({emo:e,p:probArray[i]})).sort((a,b)=>b.p-a.p);
  let html = `<div class="bars">`;
  for (const {emo,p} of pairs){
    html += `<div>${emo}</div><div class="bar"><div class="fill" style="width:${(p*100).toFixed(1)}%"></div></div>`;
  }
  html += `</div><div class="top">Top: <b>${pairs[0].emo}</b> (${(pairs[0].p*100).toFixed(1)}%)</div>`;
  els.predOut.innerHTML = html;
}

// ---------- Buttons ----------
els.btnLoad.onclick = async () => {
  try{
    if (!els.trainFile?.files?.[0]){ log('Select train.txt'); return; }
    state.train = await readTxtFile(els.trainFile.files[0]);
    state.val   = els.valFile?.files?.[0]  ? await readTxtFile(els.valFile.files[0])  : [];
    state.test  = els.testFile?.files?.[0] ? await readTxtFile(els.testFile.files[0]) : [];

    if (!state.train.length){ log('ERROR: train.txt parsed 0 lines. Ensure format is "text;label".'); return; }

    inferLabelsFromTrain(state.train);

    const keep = it => label2id[it.label] !== undefined;
    state.train = state.train.filter(keep);
    state.val   = state.val.filter(keep);
    state.test  = state.test.filter(keep);

    state.vocabSize = parseInt(els.vocabSize?.value,10) || 10000;
    state.maxLen    = parseInt(els.maxLen?.value,10)    || 30;

    state.tokenizer = buildTokenizer(state.train.map(d=>d.text), state.vocabSize);
    state.trainT = toXY(state.train, state.tokenizer, state.maxLen);
    state.valT   = state.val.length  ? toXY(state.val,  state.tokenizer, state.maxLen) : null;
    state.testT  = state.test.length ? toXY(state.test, state.tokenizer, state.maxLen) : null;

    els.dataStatus.textContent =
      `Loaded: train=${state.train.length}, val=${state.val.length}, test=${state.test.length} • labels=[${LABELS.join(', ')}] • vocab=${state.vocabSize} • maxLen=${state.maxLen}`;
    log(`TFJS ${tf?.version?.tfjs || 'unknown'} | Data loaded.`);
  }catch(e){ log('ERROR loading: '+e.message); console.error(e); }
};

els.btnTrain.onclick = async () => {
  try{
    if (!state.trainT){ log('Load data first.'); return; }
    if (state.model){ state.model.dispose(); state.model = null; }

    const epochs = parseInt(els.epochs?.value,10) || 8;
    const batch  = parseInt(els.batch?.value,10)  || 64;

    state.model = buildModel(state.vocabSize, state.maxLen, LABELS.length);

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
  }catch(e){ log('ERROR training: '+e.message); console.error(e); }
};

els.btnEval.onclick = async () => {
  try{
    if (!state.model){ log('Train first.'); return; }
    const set = state.testT || state.valT || state.trainT;
    if (!set){ log('No set to evaluate.'); return; }

    const probs = state.model.predict(set.xs);
    const yhat  = probs.argMax(-1);
    const ypred = Array.from(await yhat.data());
    const ytrue = Array.from(await set.ys.data());
    const acc = ypred.filter((p,i)=>p===ytrue[i]).length / ytrue.length;

    log(`Accuracy: ${(acc*100).toFixed(2)}% on ${ytrue.length} samples.`);
    showConfusionMatrix(ytrue, ypred);

    // show 5 examples
    for (let i=0;i<Math.min(5, ytrue.length); i++){
      const j = Math.floor(Math.random()*ytrue.length);
      log(`Ex ${i+1}: "${state.test?.[j]?.text || state.val?.[j]?.text || state.train[j].text}" | True=${LABELS[ytrue[j]]} | Pred=${LABELS[ypred[j]]}`);
    }
    probs.dispose(); yhat.dispose();
  }catch(e){ log('ERROR eval: '+e.message); console.error(e); }
};

els.btnPredict.onclick = async () => {
  try{
    if (!state.model || !state.tokenizer){ log('Load & train model first.'); return; }
    const txt = (els.predictText?.value || '').trim();
    if (!txt) return;
    const seq = state.tokenizer.toSeq(txt, state.maxLen);
    const xs  = tf.tensor2d([seq], [1,state.maxLen], 'int32');
    const p   = state.model.predict(xs);
    const arr = await p.data();
    renderPrediction(Array.from(arr));
    p.dispose(); xs.dispose();
  }catch(e){ log('ERROR predict: '+e.message); console.error(e); }
};

els.btnReset && (els.btnReset.onclick = () => {
  try{
    if (state.model){ state.model.dispose(); state.model = null; }
    ['train','val','test','trainT','valT','testT'].forEach(k => state[k]=null);
    state.tokenizer=null; LABELS=[]; label2id=Object.create(null);
    if (els.logs) els.logs.textContent='';
    if (els.results) els.results.innerHTML='';
    if (els.predOut) els.predOut.innerHTML='';
    if (els.dataStatus) els.dataStatus.textContent='No data loaded';
    try { tfvis.visor().close(); } catch(_){}
    log('Reset complete.');
  }catch(e){ log('ERROR reset: '+e.message); }
});

// Optional: saving/loading model + tokenizer
els.btnSaveModel && (els.btnSaveModel.onclick = async () => {
  if (!state.model) return log('No model to save.');
  await state.model.save('downloads://emotion_mlp_mlp_only');
  log('Model saved.');
});
els.btnSaveTok && (els.btnSaveTok.onclick = () => {
  if (!state.tokenizer) return log('No tokenizer to save.');
  const blob = new Blob([JSON.stringify({
    wordIndex: state.tokenizer.wordIndex, maxLen: state.maxLen, vocabSize: state.vocabSize, labels: LABELS
  }, null, 2)], {type:'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'tokenizer.json'; a.click(); URL.revokeObjectURL(a.href);
  log('Tokenizer saved.');
});
els.btnLoadModel && (els.btnLoadModel.onclick = async () => {
  try{
    const jf = els.modelJson?.files?.[0], wf = els.modelWeights?.files?.[0], tfj = els.tokJson?.files?.[0];
    if (!jf || !wf || !tfj) return log('Pick model.json, weights.bin, tokenizer.json');
    if (state.model) state.model.dispose();
    state.model = await tf.loadLayersModel(tf.io.browserFiles([jf,wf]));
    const tok = JSON.parse(await tfj.text());
    state.maxLen = tok.maxLen || 30; state.vocabSize = tok.vocabSize || 10000;
    LABELS = Array.isArray(tok.labels) ? tok.labels.slice() : LABELS;
    label2id = Object.create(null); LABELS.forEach((l,i)=>label2id[l]=i);
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
  }catch(e){ log('ERROR loading model: '+e.message); console.error(e); }
});
