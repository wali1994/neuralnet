async function analyzeText() {
    const text = document.getElementById('moodText').value;  // Get user input from textarea

    // Check if the input is empty
    if (!text.trim()) {
        alert('Please describe your mood first');
        return;
    }

    try {
        // Send POST request to the Flask backend
        const response = await fetch('http://127.0.0.1:5000/analyze', {  // Update with your deployed backend URL later
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        // If the request is successful, display the mood result
        if (response.ok) {
            displayResults(data);  // Pass the response to your existing displayResults function
        } else {
            alert(data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert("Text analysis failed. Please try again.");
    }
}

function displayResults(result) {
    // Extract data from the response (mood, emoji, confidence, etc.)
    const { mood, emoji, recommendation, confidence } = result;

    // Update the UI with the results
    document.getElementById("mood-result").innerHTML = `
        <div class="alert" style="background-color: ${confidence > 80 ? '#10b981' : '#ef4444'}; color: white; font-size: 20px;">
            <span>${emoji} You're feeling ${mood}!</span>
            <p><strong>Recommendation:</strong> ${recommendation}</p>
            <p><strong>Confidence:</strong> ${confidence}%</p>
        </div>
    `;
}
