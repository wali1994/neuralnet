// Function to analyze text mood
async function analyzeText() {
    const text = document.getElementById('text-input').value;  // Get the input text

    // Check if text is empty
    if (!text.trim()) {
        alert('Please describe your mood first');
        return;
    }

    try {
        // Send POST request to Flask backend (make sure to update the backend URL if deployed)
        const response = await fetch('https://your-backend-url.onrender.com/analyze', {  // Use your deployed backend URL
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })  // Send text in JSON format
        });

        const data = await response.json();  // Get the response as JSON

        // If the request is successful, display the mood result
        if (response.ok) {
            displayResults(data);  // Pass the response to displayResults()
        } else {
            alert(data.error);  // Show error message
        }
    } catch (error) {
        console.error('Error:', error);
        alert("Text analysis failed. Please try again.");
    }
}

// Function to display the mood result
function displayResults(result) {
    const { mood, emoji, recommendation, confidence } = result;

    // Display the mood analysis results
    document.getElementById("mood-result").innerHTML = `
        <div class="alert" style="background-color: ${confidence > 80 ? '#10b981' : '#ef4444'}; color: white; font-size: 20px;">
            <span>${emoji} You're feeling ${mood}!</span>
            <p><strong>Recommendation:</strong> ${recommendation}</p>
            <p><strong>Confidence:</strong> ${confidence}%</p>
        </div>
    `;
}

// Function to analyze voice mood (for the recorded audio)
async function analyzeVoice() {
    const audioBlob = document.getElementById('recorded-audio').src; // Get the audio source

    // Ensure the audio has been recorded
    if (!audioBlob) {
        alert("Please record your voice first.");
        return;
    }

    try {
        // Send POST request to Flask backend (use your deployed backend URL)
        const response = await fetch('https://your-backend-url.onrender.com/analyze', {  // Use your deployed backend URL
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ audioBlob: audioBlob })  // Send the audio for analysis
        });

        const data = await response.json();  // Get the response as JSON

        // If the request is successful, display the mood result
        if (response.ok) {
            displayResults(data);  // Pass the response to displayResults()
        } else {
            alert(data.error);  // Show error message
        }
    } catch (error) {
        console.error('Error:', error);
        alert("Voice analysis failed. Please try again.");
    }
}

// Function to start voice recording
function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }) // Request microphone access
        .then(stream => {
            const mediaRecorder = new MediaRecorder(stream);  // Create MediaRecorder instance
            const audioChunks = [];  // Array to store audio data

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data); // Push audio data
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });  // Create audio blob
                const audioUrl = URL.createObjectURL(audioBlob);  // Convert to URL for playback
                document.getElementById("recorded-audio").src = audioUrl;  // Set the audio source
                document.getElementById("analyze-recording-btn").disabled = false;  // Enable Analyze button after recording
            };

            mediaRecorder.start(); // Start recording

            // Show recording UI
            document.getElementById("start-record-btn").style.display = "none";
            document.getElementById("stop-record-btn").style.display = "inline-block";
            document.getElementById("recording-timer").style.display = "block";  // Show recording timer

            // Stop the recording after a certain duration (optional)
            setTimeout(() => {
                mediaRecorder.stop();
            }, 5000);  // Stop recording after 5 seconds (you can change this value)
        })
        .catch(error => {
            console.error("Error accessing the microphone:", error);
            alert("Microphone access denied. Please enable microphone permissions.");
        });
}

// Function to stop recording voice
function stopRecording() {
    // Hide stop button and show start button again
    document.getElementById("stop-record-btn").style.display = "none";
    document.getElementById("start-record-btn").style.display = "inline-block";
    document.getElementById("recording-timer").style.display = "none"; // Hide the timer
}

// Function to clear the recorded audio
function clearRecording() {
    document.getElementById("recorded-audio").src = ""; // Clear audio playback
    document.getElementById("analyze-recording-btn").disabled = true; // Disable analyze button
}
