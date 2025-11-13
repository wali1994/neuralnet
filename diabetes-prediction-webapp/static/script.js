document.getElementById("predictForm").addEventListener("submit", function (event) {
    event.preventDefault();

    const data = {
        gender: document.getElementById("gender").value,
        age: document.getElementById("age").value,
        hypertension: document.getElementById("hypertension").value,
        heart_disease: document.getElementById("heart_disease").value,
        smoking_history: document.getElementById("smoking_history").value,
        bmi: document.getElementById("bmi").value,
        HbA1c_level: document.getElementById("HbA1c_level").value,
        blood_glucose_level: document.getElementById("blood_glucose_level").value
    };

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
        .then(response => response.json())
        .then(res => {
            document.getElementById("result").textContent = res.result;
            document.getElementById("probability").textContent =
                "Predicted probability: " + res.probability.toFixed(3);
            document.getElementById("comment").textContent = res.comment;
        })
        .catch(err => {
            console.error(err);
            document.getElementById("result").textContent = "Error while predicting.";
            document.getElementById("probability").textContent = "";
            document.getElementById("comment").textContent = "";
        });
});
