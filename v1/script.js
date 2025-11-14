// Global state
let rawRows = [];
let featureOrder = ["genderNum", "age", "hypertension", "heart_disease", "smokingCode", "bmi", "HbA1c_level", "blood_glucose_level"];
let scaler = { mean: [], std: [] };
let logModel = null;
let nnModel = null;
let modelsReady = false;

const smokingMap = {
    "never": 0,
    "No Info": 1,
    "former": 2,
    "current": 3,
    "not current": 4,
    "ever": 5
};

const datasetInput = document.getElementById("datasetInput");
const trainButton = document.getElementById("trainButton");
const edaDiv = document.getElementById("eda");
const metricsDiv = document.getElementById("metrics");
const predictForm = document.getElementById("predictForm");
const predictButton = document.getElementById("predictButton");
const predictionOutput = document.getElementById("predictionOutput");

// 1. Load dataset
datasetInput.addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (!file) return;

    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function (results) {
            rawRows = cleanRows(results.data);
            showEDA(rawRows);
            trainButton.disabled = rawRows.length === 0;
        }
    });
});

// Clean and basic preprocessing for rows
function cleanRows(rows) {
    const cleaned = [];
    for (const r of rows) {
        if (r.diabetes !== 0 && r.diabetes !== 1) continue;
        if (r.gender !== "Male" && r.gender !== "Female") continue;
        const smoking = r.smoking_history;
        if (!(smoking in smokingMap)) continue;

        const vals = [
            r.age,
            r.hypertension,
            r.heart_disease,
            r.bmi,
            r.HbA1c_level,
            r.blood_glucose_level
        ];
        if (vals.some(v => v === null || v === undefined || Number.isNaN(v))) continue;

        const genderNum = r.gender === "Male" ? 1 : 0;
        const smokingCode = smokingMap[smoking];

        cleaned.push({
            genderNum,
            age: r.age,
            hypertension: r.hypertension,
            heart_disease: r.heart_disease,
            smokingCode,
            bmi: r.bmi,
            HbA1c_level: r.HbA1c_level,
            blood_glucose_level: r.blood_glucose_level,
            diabetes: r.diabetes
        });
    }
    return cleaned;
}

function showEDA(rows) {
    if (rows.length === 0) {
        edaDiv.innerHTML = "<p>No valid rows found in dataset.</p>";
        return;
    }
    const n = rows.length;
    let positives = 0;
    let ageSum = 0;
    let bmiSum = 0;
    for (const r of rows) {
        if (r.diabetes === 1) positives++;
        ageSum += r.age;
        bmiSum += r.bmi;
    }
    const neg = n - positives;
    const posPct = (positives / n * 100).toFixed(2);
    const negPct = (neg / n * 100).toFixed(2);
    const meanAge = (ageSum / n).toFixed(1);
    const meanBmi = (bmiSum / n).toFixed(1);

    edaDiv.innerHTML = `
        <p><strong>Rows:</strong> ${n}</p>
        <p><strong>Class distribution:</strong> 0 → ${neg} (${negPct}%), 1 → ${positives} (${posPct}%)</p>
        <p><strong>Mean age:</strong> ${meanAge}</p>
        <p><strong>Mean BMI:</strong> ${meanBmi}</p>
    `;
}

// 2. Train models
trainButton.addEventListener("click", async function () {
    if (rawRows.length === 0) return;
    trainButton.disabled = true;
    trainButton.textContent = "Training...";

    try {
        const { XTrain, yTrain, XTest, yTest } = prepareTensors(rawRows);
        const inputDim = featureOrder.length;

        // Logistic regression (no hidden layer)
        logModel = tf.sequential();
        logModel.add(tf.layers.dense({
            units: 1,
            activation: "sigmoid",
            inputShape: [inputDim]
        }));
        logModel.compile({
            optimizer: tf.train.adam(0.01),
            loss: "binaryCrossentropy",
            metrics: ["accuracy"]
        });

        await logModel.fit(XTrain, yTrain, {
            epochs: 20,
            batchSize: 64,
            shuffle: true,
            verbose: 0
        });

        const logEval = logModel.evaluate(XTest, yTest);
        const logLoss = (await logEval[0].data())[0];
        const logAcc = (await logEval[1].data())[0];

        // Neural network with hidden layer
        nnModel = tf.sequential();
        nnModel.add(tf.layers.dense({
            units: 32,
            activation: "relu",
            inputShape: [inputDim]
        }));
        nnModel.add(tf.layers.dense({
            units: 16,
            activation: "relu"
        }));
        nnModel.add(tf.layers.dense({
            units: 1,
            activation: "sigmoid"
        }));
        nnModel.compile({
            optimizer: tf.train.adam(0.005),
            loss: "binaryCrossentropy",
            metrics: ["accuracy"]
        });

        await nnModel.fit(XTrain, yTrain, {
            epochs: 25,
            batchSize: 64,
            shuffle: true,
            verbose: 0
        });

        const nnEval = nnModel.evaluate(XTest, yTest);
        const nnLoss = (await nnEval[0].data())[0];
        const nnAcc = (await nnEval[1].data())[0];

        metricsDiv.innerHTML = `
            <p><strong>Logistic regression:</strong> accuracy ${(logAcc * 100).toFixed(2)}%, loss ${logLoss.toFixed(4)}</p>
            <p><strong>Neural network:</strong> accuracy ${(nnAcc * 100).toFixed(2)}%, loss ${nnLoss.toFixed(4)}</p>
        `;

        modelsReady = true;
        predictButton.disabled = false;
    } catch (err) {
        console.error(err);
        metricsDiv.innerHTML = "<p>Error while training models. Check console.</p>";
    } finally {
        trainButton.textContent = "Train models";
    }
});

function prepareTensors(rows) {
    // maybe sample to keep training light
    const maxRows = 5000;
    let data = rows;
    if (rows.length > maxRows) {
        data = shuffle(rows).slice(0, maxRows);
    }

    const X = [];
    const y = [];
    for (const r of data) {
        X.push([
            r.genderNum,
            r.age,
            r.hypertension,
            r.heart_disease,
            r.smokingCode,
            r.bmi,
            r.HbA1c_level,
            r.blood_glucose_level
        ]);
        y.push(r.diabetes);
    }

    const n = X.length;
    const indices = [...Array(n).keys()];
    shuffle(indices);

    const testRatio = 0.2;
    const testSize = Math.floor(n * testRatio);
    const testIdx = new Set(indices.slice(0, testSize));

    const XTrain = [];
    const yTrain = [];
    const XTest = [];
    const yTest = [];

    for (let i = 0; i < n; i++) {
        if (testIdx.has(i)) {
            XTest.push(X[i]);
            yTest.push(y[i]);
        } else {
            XTrain.push(X[i]);
            yTrain.push(y[i]);
        }
    }

    // compute scaler on train
    scaler = computeScaler(XTrain);

    const XTrainScaled = applyScaler(XTrain, scaler);
    const XTestScaled = applyScaler(XTest, scaler);

    const XTrainTensor = tf.tensor2d(XTrainScaled);
    const yTrainTensor = tf.tensor2d(yTrain, [yTrain.length, 1]);
    const XTestTensor = tf.tensor2d(XTestScaled);
    const yTestTensor = tf.tensor2d(yTest, [yTest.length, 1]);

    return { XTrain: XTrainTensor, yTrain: yTrainTensor, XTest: XTestTensor, yTest: yTestTensor };
}

function computeScaler(X) {
    const n = X.length;
    const d = X[0].length;
    const mean = new Array(d).fill(0);
    const std = new Array(d).fill(0);

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
            mean[j] += X[i][j];
        }
    }
    for (let j = 0; j < d; j++) {
        mean[j] /= n;
    }
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
            const diff = X[i][j] - mean[j];
            std[j] += diff * diff;
        }
    }
    for (let j = 0; j < d; j++) {
        std[j] = Math.sqrt(std[j] / n) || 1;
    }
    return { mean, std };
}

function applyScaler(X, scaler) {
    const n = X.length;
    const d = X[0].length;
    const out = new Array(n);

    for (let i = 0; i < n; i++) {
        const row = new Array(d);
        for (let j = 0; j < d; j++) {
            row[j] = (X[i][j] - scaler.mean[j]) / scaler.std[j];
        }
        out[i] = row;
    }
    return out;
}

function shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

// 3. User prediction
predictForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    if (!modelsReady || !nnModel) {
        predictionOutput.innerHTML = "<p>Train the models first.</p>";
        return;
    }

    const genderVal = document.getElementById("gender").value;
    const age = parseFloat(document.getElementById("age").value);
    const hypertension = parseInt(document.getElementById("hypertension").value);
    const heart = parseInt(document.getElementById("heart_disease").value);
    const smokingVal = document.getElementById("smoking_history").value;
    const bmi = parseFloat(document.getElementById("bmi").value);
    const hba1c = parseFloat(document.getElementById("hba1c").value);
    const glucose = parseFloat(document.getElementById("glucose").value);

    const genderNum = genderVal === "Male" ? 1 : 0;
    const smokingCode = smokingMap[smokingVal];

    const feats = [
        genderNum,
        age,
        hypertension,
        heart,
        smokingCode,
        bmi,
        hba1c,
        glucose
    ];

    const scaled = applyScaler([feats], scaler)[0];
    const xTensor = tf.tensor2d([scaled]);
    const prob = (await nnModel.predict(xTensor).data())[0];
    xTensor.dispose();

    const label = prob >= 0.5 ? "Positive" : "Negative";
    const msg = label === "Positive"
        ? "You may be at higher risk of diabetes. Please consult a doctor for medical advice."
        : "Your predicted diabetes risk is low based on this model. This does not replace medical tests.";

    predictionOutput.innerHTML = `
        <p><strong>Prediction:</strong> Diabetes risk ${label}</p>
        <p><strong>Probability:</strong> ${(prob * 100).toFixed(2)}%</p>
        <p>${msg}</p>
    `;
});
