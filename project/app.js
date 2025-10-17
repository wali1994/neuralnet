// app.js
/* Emotion Recognition: TF.js MLP with on-page tokenizer + training.
   Supports Kaggle Emotions dataset (train/val/test .txt with "text<TAB>label"). */

const EMOTIONS = ["joy","sadness","anger","fear","love","surprise"]; // expected labels
const state = {
  train: null, val: null, test: null,
  tokenizer: null, model: null,
  maxLen: 30, vocabSize: 10000,
};

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
  charts: document.getElementById('charts'),
  results: document.getElementById('results'),
  predOut: document.getElementById('predOut'),
};

function log(msg){ els.logs.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`; els.logs.scrollTop = els.logs.scrollHeight; }

function tokenizeBasic(s){
  // lowercase, keep apostrophes/numbers, split by non-word
  return s.toLowerCase().replace(/[^\w\s’']/g,' ').split(/\s+/).filter(Boolean);
}

async function readTxtFile(file){
  const txt = await file.text();
  const lines = txt.split(/\r?\n/).filter(Boolean);
  const data = [];
  for (const line of lines){
    const parts = line.split('\t');
    if (parts.length < 2) continue;
    const text = parts[0].trim();
    const label = parts[1].trim();
    if (!text) continue;
    data.push({text, label});
  }
  return data;
}

function buildTokenizer(samples, vocabSize){
  const freq = new Map();
  for (const s of samples){
    const toks = tokenizeBasic(s);
    toks.forEach(t => freq.set(t,(freq.get(t)||0)+1));
  }
  // sort by frequency
  const sorted = [...freq.entries()].sort((a,b)=>b[1]-a[1]).slice(0, vocabSize-2);
  const wordIndex = Object.create(null); // 0=pad, 1=OOV
  let idx = 2;
  for (const [w] of sorted){ wordIndex[w] = idx++; }
  return {
    wordIndex,
    toSeq(text, maxLen){
      const toks = tokenizeBasic(text);
      const arr = new Array(maxLen).fill(0);
      for (let i=0;i<Math.min(toks.length,maxLen);i++){
        const id = wordIndex[toks[i]] || 1; // OOV=1
        arr[i] = id;
      }
      return arr;
    }
  };
}

function toXY(items, tokenizer, maxLen){
  const X = new Float32Array(items.length * maxLen);
  const y = new Int32Array(items.length);
  for (let i=0;i<items.length;i++){
    const seq = tokenizer.toSeq(items[i].text, maxLen);
    X.set(seq, i*maxLen);
    const cls = Math.max(0, EMOTIONS.indexOf(items[i].label));
    y[i] = cls;
  }
  const xs = tf.tensor2d(X, [items.length, maxLen], 'int32');
  const ys = tf.tensor1d(y, 'int32');
  return { xs, ys };
}

function buildModel(vocabSize, maxLen, numClasses){
  const input = tf.input({shape:[maxLen]});
  let x = tf.layers.embedding({inputDim:vocabSize, outputDim:128, inputLength:maxLen}).apply(input);
  x = tf.layers.globalAveragePooling1d().apply(x);
  x = tf.layers.dense({units:128, activation:'relu'}).apply(x);
  x = tf.layers.dropout({rate:0.3}).apply(x);
  x = tf.layers.dense({units:64, activation:'relu'}).apply(x);
  const out = tf.layers.dense({units:numClasses, activation:'softmax'}).apply(x);
  const model = tf.model({inputs:input, outputs:out});
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });
  return model;
}

function showConfusionMatrix(yTrue, yPred){
  const K = EMOTIONS.length;
  const M = Array.from({length:K}, ()=>Array(K).fill(0));
  for (let i=0;i<yTrue.length;i++) M[yTrue[i]][yPred[i]]++;
  // Render
  let html = `<table class="cm"><tr><th></th>${EMOTIONS.map(e=>`<th>${e}</th>`).join('')}</tr>`;
  for (let r=0;r<K;r++){
    html += `<tr><th>${EMOTIONS[r]}</th>${M[r].map(n=>`<td>${n}</td>`).join('')}</tr>`;
  }
  html += `</table>`;
  els.results.innerHTML = html;
}

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

/* ----------------- Wire up UI ----------------- */
els.btnLoad.onclick = async () => {
  try{
    if (!els.trainFile.files[0]){ log('Select train.txt'); return; }
    state.train = await readTxtFile(els.trainFile.files[0]);
    if (els.valFile.files[0]) state.val = await readTxtFile(els.valFile.files[0]);
    if (els.testFile.files[0]) state.test = await readTxtFile(els.testFile.files[0]);

    // tokenizer from training text only
    state.vocabSize = parseInt(els.vocabSize.value,10);
    state.maxLen = parseInt(els.maxLen.value,10);
    state.tokenizer = buildTokenizer(state.train.map(d=>d.text), state.vocabSize);

    // Build tensors
    const tr = toXY(state.train, state.tokenizer, state.maxLen);
    const va = state.val ? toXY(state.val, state.tokenizer, state.maxLen) : null;
    const te = state.test ? toXY(state.test, state.tokenizer, state.maxLen) : null;
    state.trainT = tr; state.valT = va; state.testT = te;

    els.dataStatus.textContent = `Loaded: train=${state.train.length}${state.val?`, val=${state.val.length}`:''}${state.test?`, test=${state.test.length}`:''} • vocab=${state.vocabSize} • maxLen=${state.maxLen}`;
    log('Data loaded and tokenized.');
  }catch(err){ log('ERROR loading: '+err.message); console.error(err); }
};

els.btnTrain.onclick = async () => {
  try{
    if (!state.trainT) return log('Load data first.');
    if (state.model){ state.model.dispose(); state.model=null; }

    const epochs = parseInt(els.epochs.value,10);
    const batch = parseInt(els.batch.value,10);
    state.model = buildModel(state.vocabSize, state.maxLen, EMOTIONS.length);

    const callbacks = tfvis.show.fitCallbacks(
      {name:'Training', tab:'Charts'},
      ['loss','val_loss','acc','val_acc'],
      {callbacks:['onEpochEnd']}
    );

    log(`Training… epochs=${epochs}, batch=${batch}`);
    await state.model.fit(state.trainT.xs, state.trainT.ys, {
      epochs, batchSize: batch,
      validationData: state.valT ? [state.valT.xs, state.valT.ys] : null,
      shuffle:true,
      callbacks
    });
    log('Training complete.');
  }catch(err){ log('ERROR training: '+err.message); console.error(err); }
};

els.btnEval.onclick = async () => {
  try{
    if (!state.model) return log('Train or load a model first.');
    if (!state.testT && !state.valT) return log('No test/val set loaded.');
    const set = state.testT || state.valT;
    const probs = state.model.predict(set.xs);
    const yhat = probs.argMax(-1);
    const ypred = Array.from((await yhat.data()));
    const ytrue = Array.from((await set.ys.data()));
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
    const xs = tf.tensor2d([seq], [1,state.maxLen], 'int32');
    const p = state.model.predict(xs);
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
  const blob = new Blob([JSON.stringify({wordIndex: state.tokenizer.wordIndex, maxLen: state.maxLen, vocabSize: state.vocabSize}, null, 2)], {type:'application/json'});
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
    state.model = await tf.loadLayersModel(tf.io.browserFiles([jf,wf]));
    const tok = JSON.parse(await tfj.text());
    state.maxLen = tok.maxLen || state.maxLen;
    state.vocabSize = tok.vocabSize || state.vocabSize;
    // Recreate tokenizer object
    state.tokenizer = {
      wordIndex: tok.wordIndex,
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
    if (state.model){ state.model.dispose(); state.model=null; }
    ['train','val','test','trainT','valT','testT'].forEach(k => state[k]=null);
    state.tokenizer=null;
    els.logs.textContent=''; els.results.innerHTML=''; els.predOut.innerHTML='';
    els.dataStatus.textContent='No data loaded';
    tfvis.visor().close();
    log('Reset done.');
  }catch(e){ log('ERROR reset: '+e.message); }
};

