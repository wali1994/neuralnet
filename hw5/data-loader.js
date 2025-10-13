// data-loader.js
class MNISTDataLoader {
  constructor() {
    this.trainData = null;
    this.testData = null;
  }

  async loadCSVFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const text = e.target.result;
          const lines = text.split(/\r?\n/).filter(l => l.trim().length);
          const labels = [];
          const pixels = [];
          for (const line of lines) {
            const vals = line.split(/[,;\s]+/).map(Number);
            if (vals.length !== 785) continue;
            labels.push(vals[0]);
            pixels.push(vals.slice(1));
          }
          if (pixels.length === 0) return reject(new Error('No valid rows in CSV'));

          const xs = tf.tidy(() => tf.tensor2d(pixels).div(255).reshape([pixels.length,28,28,1]));
          const ys = tf.tidy(() => tf.oneHot(labels, 10));
          resolve({ xs, ys, count: pixels.length });
        } catch (err) { reject(err); }
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  }

  async loadTrainFromFiles(file){ this.trainData = await this.loadCSVFile(file); return this.trainData; }
  async loadTestFromFiles(file){ this.testData = await this.loadCSVFile(file); return this.testData; }

  splitTrainVal(xs, valRatio=0.1){
    return tf.tidy(() => {
      const n = xs.shape[0], nVal = Math.floor(n*valRatio), nTr = n - nVal;
      const trainXs = xs.slice([0,0,0,0],[nTr,28,28,1]);
      const valXs   = xs.slice([nTr,0,0,0],[nVal,28,28,1]);
      return { trainXs, valXs };
    });
  }

  getRandomTestBatch(xs, k=5){
    return tf.tidy(() => {
      const idx = tf.util.createShuffledIndices(xs.shape[0]).slice(0,k);
      const batchXs = tf.gather(xs, idx);
      return { batchXs, indices: idx };
    });
  }

  // Additive Gaussian noise in [0,1] then clipped.
  addGaussianNoise(x, std=0.5){
    return tf.tidy(() => tf.clipByValue(x.add(tf.randomNormal(x.shape,0,std,'float32')),0,1));
  }

  draw28x28ToCanvas(t, canvas, scale=4){
    return tf.tidy(() => {
      const d = t.reshape([28,28]).mul(255).dataSync();
      const img = new ImageData(28,28);
      for(let i=0;i<784;i++){
        const v = d[i]|0;
        img.data[i*4+0]=v; img.data[i*4+1]=v; img.data[i*4+2]=v; img.data[i*4+3]=255;
      }
      const off = document.createElement('canvas'); off.width=28; off.height=28;
      off.getContext('2d').putImageData(img,0,0);
      const ctx = canvas.getContext('2d');
      canvas.width=28*scale; canvas.height=28*scale;
      ctx.imageSmoothingEnabled=false;
      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.drawImage(off,0,0,28*scale,28*scale);
    });
  }

  dispose(){
    if(this.trainData){ this.trainData.xs.dispose(); this.trainData.ys.dispose(); this.trainData=null; }
    if(this.testData){ this.testData.xs.dispose(); this.testData.ys.dispose(); this.testData=null; }
  }
}
