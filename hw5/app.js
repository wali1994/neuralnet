// app.js
class MNISTApp {
  constructor() {
    this.dataLoader = new MNISTDataLoader();
    this.model = null;
    this.isTraining = false;
    this.trainData = null;
    this.testData = null;
    this.initializeUI();
  }

  initializeUI() {
    document.getElementById('loadDataBtn').addEventListener('click', () => this.onLoadData());
    document.getElementById('trainBtn').addEventListener('click', () => this.onTrain());
    document.getElementById('evaluateBtn').addEventListener('click', () => this.onEvaluate());
    document.getElementById('testFiveBtn').addEventListener('click', () => this.onTestFive());
    document.getElementById('saveModelBtn').addEventListener('click', () => this.onSaveDownload());
    document.getElementById('loadModelBtn').addEventListener('click', () => this.onLoadFromFiles());
    document.getElementById('resetBtn').addEventListener('click', () => this.onReset());
    document.getElementById('toggleVisorBtn').addEventListener('click', () => this.toggleVisor());
  }

  // ---------- DATA ----------
  async onLoadData() {
    try {
      const trainFile = document.getElementById('trainFile').files[0];
      const testFile  = document.getElementById('testFile').files[0];
      if (!trainFile || !testFile) return this.showError('Please select both train and test CSV files');

      this.showStatus('Loading training data…');
      this.trainData = await this.dataLoader.loadTrainFromFiles(trainFile);

      this.showStatus('Loading test data…');
      this.testData  = await this.dataLoader.loadTestFromFiles(testFile);

      this.updateDataStatus(this.trainData.count, this.testData.count);
      this.showStatus('Data loaded successfully!');
    } catch (e) { this.showError(`Failed to load data: ${e.message}`); }
  }

  // ---------- MODEL ----------
  buildAutoencoder(){
    const input = tf.input({shape:[28,28,1]});
    let x = tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(input);
    x = tf.layers.maxPooling2d({poolSize:2,strides:2,padding:'same'}).apply(x);
    x = tf.layers.conv2d({filters:64,kernelSize:3,padding:'same',activation:'relu'}).apply(x);
    x = tf.layers.maxPooling2d({poolSize:2,strides:2,padding:'same'}).apply(x);

    x = tf.layers.conv2dTranspose({filters:64,kernelSize:3,strides:2,padding:'same',activation:'relu'}).apply(x);
    x = tf.layers.conv2dTranspose({filters:32,kernelSize:3,strides:2,padding:'same',activation:'relu'}).apply(x);
    const out = tf.layers.conv2d({filters:1,kernelSize:3,padding:'same',activation:'sigmoid'}).apply(x);

    const model = tf.model({inputs:input, outputs:out});
    model.compile({optimizer: tf.train.adam(), loss: 'meanSquaredError'});
    return model;
  }

  updateModelInfo(){
    const infoEl = document.getElementById('modelInfo');
    if (!this.model) {
      infoEl.innerHTML = '<h3>Model Info</h3><p>No model loaded</p>'; return;
    }
    infoEl.innerHTML = `<h3>Model Info</h3>
      <p>Type: CNN Autoencoder • Loss: MSE • Optimizer: Adam</p>
      <p>Total params: ${this.model.countParams().toLocaleString()}</p>`;
  }

  // ---------- TRAIN ----------
  async onTrain() {
    if (!this.trainData) return this.showError('Please load training data first');
    if (this.isTraining) return this.showError('Training already in progress');

    try {
      this.isTraining = true;
      const epochs = parseInt(document.getElementById('epochs').value,10) || 8;
      const batch  = parseInt(document.getElementById('batch').value,10) || 128;
      const std    = parseFloat(document.getElementById('noiseStd').value) || 0.5;

      // Split images only (targets = clean images)
      const { trainXs, valXs } = this.dataLoader.splitTrainVal(this.trainData.xs, 0.1);

      if (!this.model) { this.model = this.buildAutoencoder(); this.updateModelInfo(); }

      const noisyTrain = this.dataLoader.addGaussianNoise(trainXs, std);
      const noisyVal   = this.dataLoader.addGaussianNoise(valXs, std);

      this.showStatus(`Training… epochs=${epochs}, batch=${batch}, noise σ=${std}`);
      const hist = await this.model.fit(noisyTrain, trainXs, {
        epochs, batchSize: batch, shuffle:true, validationData:[noisyVal, valXs],
        callbacks: tfvis.show.fitCallbacks({name:'Training Performance',tab:'Charts'}, ['loss','val_loss'], {callbacks:['onEpochEnd']})
      });

      const best = Math.min(...hist.history.val_loss);
      this.showStatus(`Training done. Best val_loss: ${best.toFixed(4)}`);

      noisyTrain.dispose(); noisyVal.dispose(); trainXs.dispose(); valXs.dispose();
    } catch (e) {
      this.showError(`Training failed: ${e.message}`);
    } finally {
      this.isTraining = false;
    }
  }

  // ---------- EVALUATE ----------
  async onEvaluate() {
    if (!this.model) return this.showError('No model available. Train or load a model first.');
    if (!this.testData) return this.showError('No test data available');

    const std = parseFloat(document.getElementById('noiseStd').value) || 0.5;

    tf.tidy(() => {
      const noisy = this.dataLoader.addGaussianNoise(this.testData.xs, std);
      const recon = this.model.predict(noisy);
      const mse = tf.losses.meanSquaredError(this.testData.xs, recon).mean();
      mse.data().then(v => this.showStatus(`Test MSE @ σ=${std}: ${v[0].toFixed(6)}`));
    });
  }

  // ---------- PREVIEW ----------
  async onTestFive() {
    if (!this.model || !this.testData) return this.showError('Please load both model and test data first');

    const std = parseFloat(document.getElementById('noiseStd').value) || 0.5;
    const { batchXs } = this.dataLoader.getRandomTestBatch(this.testData.xs, 5);
    const noisy = this.dataLoader.addGaussianNoise(batchXs, std);
    const recon = this.model.predict(noisy);

    await this.renderTriples(noisy, recon, batchXs);

    noisy.dispose(); recon.dispose(); batchXs.dispose();
  }

  async renderTriples(noisy, recon, clean){
    const container = document.getElementById('previewContainer');
    container.innerHTML = '';
    const k = noisy.shape[0];

    for(let i=0;i<k;i++){
      const col = document.createElement('div'); col.className='preview-item';

      const c1 = document.createElement('canvas');
      const c2 = document.createElement('canvas');
      const c3 = document.createElement('canvas');

      const n = noisy.slice([i,0,0,0],[1,28,28,1]).squeeze();
      const r = recon.slice([i,0,0,0],[1,28,28,1]).squeeze();
      const c = clean.slice([i,0,0,0],[1,28,28,1]).squeeze();

      this.dataLoader.draw28x28ToCanvas(n, c1, 4);
      this.dataLoader.draw28x28ToCanvas(r, c2, 4);
      this.dataLoader.draw28x28ToCanvas(c, c3, 4);

      col.appendChild(c1);
      col.appendChild(c2);
      col.appendChild(c3);
      container.appendChild(col);

      n.dispose(); r.dispose(); c.dispose();
    }
  }

  // ---------- SAVE / LOAD ----------
  async onSaveDownload() {
    if (!this.model) return this.showError('No model to save');
    await this.model.save('downloads://mnist_denoiser');
    this.showStatus('Model saved (downloaded).');
  }

  async onLoadFromFiles() {
    const jsonFile = document.getElementById('modelJsonFile').files[0];
    const weightsFile = document.getElementById('modelWeightsFile').files[0];
    if (!jsonFile || !weightsFile) return this.showError('Please select both model.json and weights.bin');

    if (this.model) this.model.dispose();
    this.model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
    this.updateModelInfo();
    this.showStatus('Model loaded successfully!');
  }

  // ---------- UTIL ----------
  onReset() {
    if (this.model) { this.model.dispose(); this.model = null; }
    this.dataLoader.dispose();
    this.trainData = null; this.testData = null;
    this.updateDataStatus(0,0); this.updateModelInfo();
    this.clearPreview(); this.showStatus('Reset completed');
  }

  toggleVisor(){ tfvis.visor().toggle(); }

  clearPreview(){ document.getElementById('previewContainer').innerHTML = ''; }

  updateDataStatus(trainCount, testCount){
    const el = document.getElementById('dataStatus');
    el.innerHTML = `<h3>Data Status</h3><p>Train samples: ${trainCount}</p><p>Test samples: ${testCount}</p>`;
  }

  showStatus(message){
    const logs = document.getElementById('trainingLogs');
    const div = document.createElement('div');
    div.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logs.appendChild(div); logs.scrollTop = logs.scrollHeight;
  }

  showError(msg){ this.showStatus(`ERROR: ${msg}`); console.error(msg); }
}

document.addEventListener('DOMContentLoaded', () => new MNISTApp());
