// app.js — Emotion Recognition with CNN + MLP

const els = {
  trainFile: document.getElementById('trainFile'),
  valFile: document.getElementById('valFile'),
  testFile: document.getElementById('testFile'),
  btnLoad: document.getElementById('btnLoad'),
  btnTrain: document.getElementById('btnTrain'),
  btnEval: document.getElementById('btnEval'),
  log: document.getElementById('log'),
  chart: document.getElementById('chart'),
  dataStatus: document.getElementById('dataStatus'),
  vocabSize: document.getElementById('vocabSize'),
  maxLen: document.getElementById('maxLen')
};

let LABELS = [];
let label2id = {};
const state = {
  train: null, val: null, test: null,
  trainT: null, valT: null, testT: null,
  tokenizer: null, model: null,
  maxLen: 30, vocabSize: 10000
};

// Logging helper
function log(msg){
  console.log(msg);
  els.log.value += msg + "\n";
  els.log.scrollTop = els.log.scrollHeight;
}

// Read TXT file (text;label)
async function readTxtFile(file){
  const text = await file.text();
  const lines = text.split(/\r?\n/).map(l=>l.trim()).filter(l=>l.length>0);
  const items = [];
  for (const line of lines){
    const [textPart, label] = line.split(';');
    if (!textPart || !label) continue;
    items.push({ text: textPart.trim().toLowerCase(), label: label.trim().toLowerCase() });
  }
  return items;
}

// Tokenizer
function buildTokenizer(texts, vocabSize){
  const freq = new Map();
  for (const t of texts){
    for (const w of t.split(/\s+/)){
      freq.set(w, (freq.get(w) || 0) + 1);
    }
  }
  const sorted = Array.from(freq.entries()).sort((a,b)=>b[1]-a[1]).slice(0,vocabSize-2);
  const word2idx = Object.create(null);
  word2idx['<PAD>'] = 0; word2idx['<UNK>'] = 1;
  sorted.forEach(([w],i)=>{ word2idx[w] = i+2; });
  return {
    word2idx,
    toSeq(text, maxLen){
      const seq = text.split(/\s+/).map(w => word2idx[w] || 1);
      if (seq.length > maxLen) return seq.slice(0, maxLen);
      while (seq.length < maxLen) seq.push(0);
      return seq;
    }
  };
}

// Infer labels dynamically
function inferLabelsFromTrain(train){
  const set = new Set(train.map(d => d.label));
  LABELS = Array.from(set).sort();
  label2id = {};
  LABELS.forEach((l,i)=>label2id[l] = i);
  log(`Labels inferred: ${LABELS.join(', ')}`);
}

// Convert to tensors
function toXY(items, tokenizer, maxLen){
  const X = new Int32Array(items.length * maxLen);
  const y = new Int32Array(items.length);
  for (let i=0;i<items.length;i++){
    const seq = tokenizer.toSeq(items[i].text, maxLen);
    X.set(seq, i*maxLen);
    y[i] = label2id[items[i].label];
  }
  const xs = tf.tensor2d(X, [items.length, maxLen], 'int32');
  const ys = tf.tensor1d(y, 'int32');
  return { xs, ys };
}

// Model: CNN + MLP (no dropout)
function buildModel(vocabSize, maxLen, numClasses){
  const input = tf.input({shape:[maxLen]});
  let x = tf.layers.embedding({inputDim:vocabSize, outputDim:128, inputLength:maxLen}).apply(input);

  const convs = [3,4,5].map(k => {
    const c = tf.layers.conv1d({filters:128, kernelSize:k, activation:'relu'}).apply(x);
    return tf.layers.globalMaxPooling1d().apply(c);
  });
  let feat = tf.layers.concatenate().apply(convs);
  feat = tf.layers.dense({units:128, activation:'relu'}).apply(feat);
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

// Load data
els.btnLoad.onclick = async () => {
  try{
    if (!els.trainFile.files[0]){ log('Select train.txt'); return; }

    state.train = await readTxtFile(els.trainFile.files[0]);
    state.val = els.valFile.files[0] ? await readTxtFile(els.valFile.files[0]) : [];
    state.test = els.testFile.files[0] ? await readTxtFile(els.testFile.files[0]) : [];

    inferLabelsFromTrain(state.train);

    const keep = it => label2id[it.label] !== undefined;
    state.train = state.train.filter(keep);
    state.val = state.val.filter(keep);
    state.test = state.test.filter(keep);

    state.vocabSize = parseInt(els.vocabSize.value,10) || 10000;
    state.maxLen = parseInt(els.maxLen.value,10) || 30;

    state.tokenizer = buildTokenizer(state.train.map(d=>d.text), state.vocabSize);
    state.trainT = toXY(state.train, state.tokenizer, state.maxLen);
    state.valT = state.val.length ? toXY(state.val, state.tokenizer, state.maxLen) : null;
    state.testT = state.test.length ? toXY(state.test, state.tokenizer, state.maxLen) : null;

    els.dataStatus.textContent =
      `Loaded: train=${state.train.length}, val=${state.val.length}, test=${state.test.length}, labels=${LABELS.join(', ')}`;
    log('Data loaded and tokenized successfully.');
  }catch(err){ log('ERROR loading: '+err.message); console.error(err); }
};

// Train model
els.btnTrain.onclick = async () => {
  try{
    if (!state.trainT) return log('Load data first!');
    const numClasses = LABELS.length;
    state.model = buildModel(state.vocabSize, state.maxLen, numClasses);
    log(state.model.summary());

    const callbacks = tfvis.show.fitCallbacks(
      { name: 'Training Progress', tab: 'Charts' },
      ['loss','val_loss','accuracy','val_accuracy'],
      { callbacks: ['onEpochEnd'] }
    );

    await state.model.fit(state.trainT.xs, state.trainT.ys, {
      epochs: 8,
      batchSize: 32,
      validationData: state.valT ? [state.valT.xs, state.valT.ys] : null,
      callbacks
    });

    log('Training complete.');
  }catch(err){ log('Training error: '+err.message); console.error(err); }
};

// Evaluate model
els.btnEval.onclick = async () => {
  try{
    if (!state.model || !state.testT) return log('Need model and test set.');
    const preds = state.model.predict(state.testT.xs);
    const yPred = preds.argMax(-1);
    const yTrue = state.testT.ys;
    const acc = (await tf.equal(yPred, yTrue).sum().array()) / yTrue.shape[0];
    log(`Test accuracy: ${(acc*100).toFixed(2)}%`);

    const sampleIdx = tf.util.createShuffledIndices(Math.min(5, state.test.length));
    const predArr = await yPred.array();
    sampleIdx.forEach(i=>{
      log(`Text: ${state.test[i].text}\nTrue: ${state.test[i].label}\nPred: ${LABELS[predArr[i]]}\n`);
    });
  }catch(err){ log('Eval error: '+err.message); console.error(err); }
};
