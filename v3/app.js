// ======== CONFIG ========

// Manual feature scaling factors to keep numbers small (no StandardScaler needed)
const SCALE = {
  age: 100,              // age / 100
  bmi: 50,               // bmi / 50
  HbA1c_level: 10,       // HbA1c / 10
  blood_glucose_level: 300 // glucose / 300
};

// Order of features for both models
const FEATURE_ORDER = [
  "gender_num",
  "hypertension",
  "heart_disease",
  "age_scaled",
  "bmi_scaled",
  "HbA1c_scaled",
  "glucose_scaled",
  "smoke_No Info",
  "smoke_current",
  "smoke_ever",
  "smoke_former",
  "smoke_never",
  "smoke_not current"
];

const SMOKING_CATEGORIES = [
  "No Info",
  "current",
  "ever",
  "former",
  "never",
  "not current"
];

// ======== STATE ========

let trainingDone = false;

const logisticModel = {
  w: new Array(FEATURE_ORDER.length).fill(0),
  b: 0
};

const nnModel = {
  hiddenSize: 8,
  W1: null, // [nFeatures x hiddenSize]
  b1: null, // [hiddenSize]
  W2: null, // [hiddenSize]
  b2: 0
};

// ======== DOM HELPERS ========

const $ = (id) => document.getElementById(id);

function setTrainingStatus(text) {
  $("training-status").textContent = text;
}

function setTrainingProgress(ratio) {
  const pct = Math.max(0, Math.min(1, ratio)) * 100;
  $("training-progress").style.width = pct.toFixed(1) + "%";
}

// ======== MATH HELPERS ========

function sigmoid(x) {
  if (x < -30) return 0;
  if (x > 30) return 1;
  return 1 / (1 + Math.exp(-x));
}

function tanh(x) {
  const e1 = Math.exp(x);
  const e2 = Math.exp(-x);
  return (e1 - e2) / (e1 + e2);
}

// ======== DATA LOADING & ENCODING ========

async function loadDataset() {
  const res = await fetch("diabetes_raw_cleaned_25k.csv");
  const text = await res.text();

  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",");
  // Expected columns:
  // gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes

  const X = [];
  const y = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const cols = line.split(",");

    const row = {};
    header.forEach((h, idx) => {
      row[h] = cols[idx];
    });

    // Encode features
    const gender_num = row.gender === "Male" ? 1 : 0;
    const age = parseFloat(row.age);
    const hypertension = parseInt(row.hypertension, 10);
    const heart_disease = parseInt(row.heart_disease, 10);
    const smoking_history = row.smoking_history;
    const bmi = parseFloat(row.bmi);
    const HbA1c_level = parseFloat(row.HbA1c_level);
    const blood_glucose_level = parseFloat(row.blood_glucose_level);
    const label = parseInt(row.diabetes, 10);

    if (
      Number.isNaN(age) ||
      Number.isNaN(bmi) ||
      Number.isNaN(HbA1c_level) ||
      Number.isNaN(blood_glucose_level)
    ) {
      continue;
    }

    // Manual scaling
    const age_scaled = age / SCALE.age;
    const bmi_scaled = bmi / SCALE.bmi;
    const HbA1c_scaled = HbA1c_level / SCALE.HbA1c_level;
    const glucose_scaled = blood_glucose_level / SCALE.blood_glucose_level;

    const smokeFlags = {
      "smoke_No Info": smoking_history === "No Info" ? 1 : 0,
      "smoke_current": smoking_history === "current" ? 1 : 0,
      "smoke_ever": smoking_history === "ever" ? 1 : 0,
      "smoke_former": smoking_history === "former" ? 1 : 0,
      "smoke_never": smoking_history === "never" ? 1 : 0,
      "smoke_not current": smoking_history === "not current" ? 1 : 0
    };

    const feat = [];

    FEATURE_ORDER.forEach((f) => {
      switch (f) {
        case "gender_num":
          feat.push(gender_num);
          break;
        case "hypertension":
          feat.push(hypertension);
          break;
        case "heart_disease":
          feat.push(heart_disease);
          break;
        case "age_scaled":
          feat.push(age_scaled);
          break;
        case "bmi_scaled":
          feat.push(bmi_scaled);
          break;
        case "HbA1c_scaled":
          feat.push(HbA1c_scaled);
          break;
        case "glucose_scaled":
          feat.push(glucose_scaled);
          break;
        default:
          feat.push(smokeFlags[f] ?? 0);
      }
    });

    X.push(feat);
    y.push(label);
  }

  return { X, y };
}

// ======== LOGISTIC REGRESSION TRAINING ========

function trainLogistic(X, y, epochs = 60, lr = 0.1, onProgress = () => {}) {
  const nSamples = X.length;
  const nFeatures = X[0].length;
  const w = new Array(nFeatures).fill(0);
  let b = 0;

  for (let ep = 0; ep < epochs; ep++) {
    const gradW = new Array(nFeatures).fill(0);
    let gradB = 0;

    for (let i = 0; i < nSamples; i++) {
      const xi = X[i];
      const yi = y[i];
      let z = b;
      for (let j = 0; j < nFeatures; j++) {
        z += w[j] * xi[j];
      }
      const p = sigmoid(z);
      const error = p - yi;
      for (let j = 0; j < nFeatures; j++) {
        gradW[j] += error * xi[j];
      }
      gradB += error;
    }

    const invN = 1 / nSamples;
    for (let j = 0; j < nFeatures; j++) {
      w[j] -= lr * gradW[j] * invN;
    }
    b -= lr * gradB * invN;

    onProgress((ep + 1) / (epochs * 2)); // logistic = half of progress bar
  }

  logisticModel.w = w;
  logisticModel.b = b;
}

// ======== NEURAL NET TRAINING ========

function initNN(nFeatures, hiddenSize) {
  nnModel.hiddenSize = hiddenSize;
  nnModel.W1 = [];
  nnModel.b1 = [];
  nnModel.W2 = [];
  nnModel.b2 = 0;

  for (let i = 0; i < nFeatures; i++) {
    nnModel.W1[i] = [];
    for (let h = 0; h < hiddenSize; h++) {
      nnModel.W1[i][h] = (Math.random() - 0.5) * 0.2;
    }
  }

  for (let h = 0; h < hiddenSize; h++) {
    nnModel.b1[h] = 0;
    nnModel.W2[h] = (Math.random() - 0.5) * 0.2;
  }
}

function trainNN(X, y, epochs = 60, lr = 0.05, onProgress = () => {}) {
  const nSamples = X.length;
  const nFeatures = X[0].length;
  const hiddenSize = nnModel.hiddenSize;

  if (!nnModel.W1) {
    initNN(nFeatures, hiddenSize);
  }

  const { W1, b1, W2 } = nnModel;
  let b2 = nnModel.b2;

  for (let ep = 0; ep < epochs; ep++) {
    const gradW1 = [];
    const gradB1 = new Array(hiddenSize).fill(0);
    const gradW2 = new Array(hiddenSize).fill(0);
    let gradB2 = 0;

    for (let i = 0; i < nFeatures; i++) {
      gradW1[i] = new Array(hiddenSize).fill(0);
    }

    for (let i = 0; i < nSamples; i++) {
      const xi = X[i];
      const yi = y[i];

      // forward
      const z1 = new Array(hiddenSize);
      const a1 = new Array(hiddenSize);
      for (let h = 0; h < hiddenSize; h++) {
        let sum = b1[h];
        for (let j = 0; j < nFeatures; j++) {
          sum += W1[j][h] * xi[j];
        }
        z1[h] = sum;
        a1[h] = tanh(sum);
      }

      let z2 = b2;
      for (let h = 0; h < hiddenSize; h++) {
        z2 += W2[h] * a1[h];
      }
      const a2 = sigmoid(z2);

      // backward
      const dL_dz2 = a2 - yi; // BCE derivative

      for (let h = 0; h < hiddenSize; h++) {
        gradW2[h] += dL_dz2 * a1[h];
      }
      gradB2 += dL_dz2;

      const dL_da1 = new Array(hiddenSize);
      const dL_dz1 = new Array(hiddenSize);

      for (let h = 0; h < hiddenSize; h++) {
        dL_da1[h] = dL_dz2 * W2[h];
        const th = a1[h];
        const sech2 = 1 - th * th; // derivative of tanh
        dL_dz1[h] = dL_da1[h] * sech2;
        gradB1[h] += dL_dz1[h];
      }

      for (let j = 0; j < nFeatures; j++) {
        for (let h = 0; h < hiddenSize; h++) {
          gradW1[j][h] += dL_dz1[h] * xi[j];
        }
      }
    }

    const invN = 1 / nSamples;

    for (let h = 0; h < hiddenSize; h++) {
      W2[h] -= lr * gradW2[h] * invN;
      b1[h] -= lr * gradB1[h] * invN;
    }
    b2 -= lr * gradB2 * invN;

    for (let j = 0; j < nFeatures; j++) {
      for (let h = 0; h < hiddenSize; h++) {
        W1[j][h] -= lr * gradW1[j][h] * invN;
      }
    }

    onProgress(0.5 + (ep + 1) / (epochs * 2)); // NN = second half of progress
  }

  nnModel.W1 = W1;
  nnModel.b1 = b1;
  nnModel.W2 = W2;
  nnModel.b2 = b2;
}

// ======== PREDICTION FUNCTIONS ========

function predictLogisticSingle(x) {
  let z = logisticModel.b;
  for (let j = 0; j < logisticModel.w.length; j++) {
    z += logisticModel.w[j] * x[j];
  }
  return sigmoid(z);
}

function predictNNSingle(x) {
  const nFeatures = x.length;
  const hiddenSize = nnModel.hiddenSize;
  const a1 = new Array(hiddenSize);

  for (let h = 0; h < hiddenSize; h++) {
    let sum = nnModel.b1[h];
    for (let j = 0; j < nFeatures; j++) {
      sum += nnModel.W1[j][h] * x[j];
    }
    a1[h] = tanh(sum);
  }

  let z2 = nnModel.b2;
  for (let h = 0; h < hiddenSize; h++) {
    z2 += nnModel.W2[h] * a1[h];
  }
  return sigmoid(z2);
}

// Build feature vector from form input
function buildFeatureFromForm() {
  const gender = $("gender").value;
  const age = parseFloat($("age").value);
  const hypertension = parseInt($("hypertension").value, 10);
  const heart_disease = parseInt($("heart_disease").value, 10);
  const smoking_history = $("smoking_history").value;
  const bmi = parseFloat($("bmi").value);
  const hba1c = parseFloat($("hba1c").value);
  const glucose = parseFloat($("glucose").value);

  const gender_num = gender === "Male" ? 1 : 0;

  const age_scaled = age / SCALE.age;
  const bmi_scaled = bmi / SCALE.bmi;
  const HbA1c_scaled = hba1c / SCALE.HbA1c_level;
  const glucose_scaled = glucose / SCALE.blood_glucose_level;

  const smokeFlags = {
    "smoke_No Info": smoking_history === "No Info" ? 1 : 0,
    "smoke_current": smoking_history === "current" ? 1 : 0,
    "smoke_ever": smoking_history === "ever" ? 1 : 0,
    "smoke_former": smoking_history === "former" ? 1 : 0,
    "smoke_never": smoking_history === "never" ? 1 : 0,
    "smoke_not current": smoking_history === "not current" ? 1 : 0
  };

  const feat = [];

  FEATURE_ORDER.forEach((f) => {
    switch (f) {
      case "gender_num":
        feat.push(gender_num);
        break;
      case "hypertension":
        feat.push(hypertension);
        break;
      case "heart_disease":
        feat.push(heart_disease);
        break;
      case "age_scaled":
        feat.push(age_scaled);
        break;
      case "bmi_scaled":
        feat.push(bmi_scaled);
        break;
      case "HbA1c_scaled":
        feat.push(HbA1c_scaled);
        break;
      case "glucose_scaled":
        feat.push(glucose_scaled);
        break;
      default:
        feat.push(smokeFlags[f] ?? 0);
    }
  });

  return {
    x: feat,
    raw: {
      age,
      bmi,
      hba1c,
      glucose,
      hypertension,
      heart_disease,
      smoking_history
    }
  };
}

// Simple rule-based comment using prediction + raw features
function buildComment(prob, raw) {
  const riskPct = (prob * 100).toFixed(1);
  const flags = [];

  if (raw.hba1c >= 6.5) flags.push("HbA1c is in the diabetic range");
  else if (raw.hba1c >= 5.7) flags.push("HbA1c is in the pre-diabetic range");

  if (raw.glucose >= 200) flags.push("blood glucose is very high");
  else if (raw.glucose >= 140) flags.push("blood glucose is elevated");

  if (raw.bmi >= 30) flags.push("BMI is in the obese range");
  else if (raw.bmi >= 25) flags.push("BMI is in the overweight range");

  if (raw.hypertension === 1) flags.push("history of hypertension");
  if (raw.heart_disease === 1) flags.push("history of heart disease");

  let riskLevel;
  if (prob < 0.25) riskLevel = "low estimated risk based on this model";
  else if (prob < 0.6) riskLevel = "moderate estimated risk";
  else riskLevel = "high estimated risk";

  let txt = `The model estimates a ${riskPct}% probability of diabetes, which this demo labels as ${riskLevel}.`;

  if (flags.length > 0) {
    txt += " Key contributing factors in your input: " + flags.join(", ") + ".";
  } else {
    txt += " Your input does not show strong risk factors inside this simple model.";
  }

  txt += " This is only a machine-learning demonstration and cannot replace professional medical screening.";

  return txt;
}

// ======== INIT ========

async function init() {
  try {
    setTrainingStatus("Loading CSV and preparing data…");
    setTrainingProgress(0);

    const { X, y } = await loadDataset();
    if (!X.length) {
      setTrainingStatus("Could not load any data from CSV.");
      return;
    }

    // Optionally subsample to speed up training
    const maxTrain = 15000;
    let trainX = X;
    let trainY = y;
    if (X.length > maxTrain) {
      const idx = [];
      for (let i = 0; i < X.length; i++) idx.push(i);
      idx.sort(() => Math.random() - 0.5);
      idx.length = maxTrain;
      trainX = idx.map((k) => X[k]);
      trainY = idx.map((k) => y[k]);
    }

    setTrainingStatus("Training logistic regression model…");
    trainLogistic(trainX, trainY, 60, 0.1, setTrainingProgress);

    setTrainingStatus("Training neural network model…");
    initNN(FEATURE_ORDER.length, nnModel.hiddenSize);
    trainNN(trainX, trainY, 60, 0.05, setTrainingProgress);

    trainingDone = true;
    setTrainingStatus("Training complete. You can now enter data and get predictions.");
    setTrainingProgress(1);
    $("predict-btn").disabled = false;
  } catch (err) {
    console.error(err);
    setTrainingStatus("Error loading or training models. Check console logs.");
  }
}

// ======== FORM HANDLER ========

function setupForm() {
  const form = $("predict-form");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (!trainingDone) {
      alert("Models are not ready yet.");
      return;
    }

    const { x, raw } = buildFeatureFromForm();

    const probLog = predictLogisticSingle(x);
    const probNN = predictNNSingle(x);
    const avgProb = (probLog + probNN) / 2;

    $("logistic-prob").textContent = (probLog * 100).toFixed(1) + "%";
    $("nn-prob").textContent = (probNN * 100).toFixed(1) + "%";

    $("logistic-label").textContent =
      probLog >= 0.5 ? "Predicted: diabetes (1)" : "Predicted: no diabetes (0)";
    $("nn-label").textContent =
      probNN >= 0.5 ? "Predicted: diabetes (1)" : "Predicted: no diabetes (0)";

    $("comment-text").textContent = buildComment(avgProb, raw);
  });
}

// ======== BOOTSTRAP ========

window.addEventListener("DOMContentLoaded", () => {
  setupForm();
  init();
});
