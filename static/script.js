// Global variables
let modelLoaded = true;
let currentPredictedDisease = ''; // Store current disease for feedback

document.addEventListener('DOMContentLoaded', function() {
    setupFileUpload();
    initializeDarkMode();
});

// dark mode
function initializeDarkMode() {
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    if (isDarkMode) {
        document.body.classList.add('dark-mode');
        updateDarkModeButton();
    }
}

function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    const isDarkMode = document.body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDarkMode);
    updateDarkModeButton();
}

function updateDarkModeButton() {
    const btn = document.getElementById('dark-mode-btn');
    const isDarkMode = document.body.classList.contains('dark-mode');
    btn.textContent = isDarkMode ? '☀️' : '🌙';
}

// SetupFile upload
function setupFileUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');

    // Drag & drop 
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
}

// Handle file upload
async function handleFile(file) {
    if (!modelLoaded) {
        showError('Model not loaded. Please train the model first.');
        return;
    }

    if (!file.type.startsWith('image/')) {
        showError('Please select an image file.');
        return;
    }

    showLoading(true);
    hideResults();

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        showLoading(false);

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Error processing image');
        }
    } catch (error) {
        showLoading(false);
        showError('Error uploading image: ' + error.message);
    }
}

// Display results
function displayResults(data) {
    if (data.image_path) {
        // If Flask returns 'static/uploads/filename', use '/static/uploads/filename'
        let imageUrl = data.image_path.replace(/^.*static[\\\/]/, '/static/');
        const imgElem = document.getElementById('uploaded-image');
        imgElem.src = imageUrl;
        imgElem.style.display = 'block';
        imgElem.alt = 'Uploaded Image';
    }
    
    // Show confidence warning if present
    if (data.confidence_warning) {
        const warningDiv = document.createElement('div');
        warningDiv.className = 'confidence-warning';
        warningDiv.innerHTML = `<strong>${data.confidence_warning}</strong>`;
        const resultsSection = document.getElementById('results-section');
        resultsSection.insertBefore(warningDiv, resultsSection.firstChild);
    }
    
    // Prediction with confidence
    const confidencePercent = (data.prediction.confidence * 100).toFixed(1);
    document.getElementById('disease-name').innerHTML = `
        ${data.prediction.disease}
        <br><span class="confidence-badge confidence-${data.prediction.confidence_level.toLowerCase()}">
            ${data.prediction.confidence_level} Confidence: ${confidencePercent}%
        </span>
    `;
    
    // Store the disease name for feedback
    currentPredictedDisease = data.prediction.disease;

    // Solution information
    const solution = data.disease_info;
    document.getElementById('symptoms').textContent = solution.symptoms;
    document.getElementById('causes').textContent = solution.causes;
    document.getElementById('prevention').textContent = solution.prevention;

    // Solutions list
    const solutionsList = document.getElementById('solutions');
    solutionsList.innerHTML = '';
    (solution.solutions || []).forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        solutionsList.appendChild(li);
    });
    
    // Show top predictions for comparison
    if (data.top_predictions && data.top_predictions.length > 1) {
        const topPredDiv = document.createElement('div');
        topPredDiv.className = 'alternative-predictions';
        topPredDiv.innerHTML = '<h3>🔍 Alternative Predictions:</h3>';
        const list = document.createElement('ul');
        data.top_predictions.forEach((pred, idx) => {
            const li = document.createElement('li');
            const confidence = (pred.confidence * 100).toFixed(1);
            li.innerHTML = `<strong>${idx + 1}.</strong> ${pred.disease} - ${confidence}%`;
            list.appendChild(li);
        });
        topPredDiv.appendChild(list);
        document.getElementById('results-section').appendChild(topPredDiv);
    }

    // Show results
    document.getElementById('results-section').style.display = 'block';
}

// Show loading
function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}

// Hide results
function hideResults() {
    document.getElementById('results-section').style.display = 'none';
}

// Show error
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.textContent = message;
    
    const container = document.querySelector('.main-content');
    const existingError = container.querySelector('.error');
    if (existingError) {
        existingError.remove();
    }
    
    container.insertBefore(errorDiv, container.firstChild);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'success';
    successDiv.textContent = message;
    
    const container = document.querySelector('.main-content');
    const existingSuccess = container.querySelector('.success');
    if (existingSuccess) {
        existingSuccess.remove();
    }
    
    container.insertBefore(successDiv, container.firstChild);
    
    setTimeout(() => {
        successDiv.remove();
    }, 5000);
}

// Feedback functionality
async function submitFeedback(isHelpful, buttonElement) {
    const feedbackMessage = document.getElementById('feedback-message');
    const feedbackButtons = document.querySelector('.feedback-buttons');
    
    // Console log for debugging
    console.log('Feedback clicked:', isHelpful, 'Button:', buttonElement);
    
    try {
        // Use the stored disease name
        const diseaseName = currentPredictedDisease;
        
        console.log('Disease name:', diseaseName);
        
        if (!diseaseName) {
            console.log('No disease name found');
            feedbackMessage.className = 'feedback-message error-feedback show';
            feedbackMessage.innerHTML = '⚠️ No disease prediction found. Please upload an image first.';
            return;
        }
        
        // Show loading state
        console.log('Showing loading state');
        feedbackMessage.className = 'feedback-message loading-feedback show';
        feedbackMessage.innerHTML = '⏳ Saving your feedback...';
        
        // Hide buttons immediately
        feedbackButtons.style.display = 'none';
        
        console.log('Sending feedback to server');
        
        // Send feedback to backend
        const response = await fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                helpful: isHelpful,
                disease: diseaseName,
                timestamp: new Date().toISOString()
            })
        });

        const data = await response.json();
        
        console.log('Response from server:', data);

        if (data.success) {
            console.log('Feedback submitted successfully');
            
            // Show success message
            feedbackMessage.className = 'feedback-message success-feedback show';
            feedbackMessage.innerHTML = isHelpful 
                ? '✅ Thank you! Your positive feedback helps us improve the model.' 
                : '❌ We appreciate your feedback. We will use it to improve our model!';
            
            // Hide the feedback buttons completely
            feedbackButtons.style.display = 'none';
            
            // Hide message after 7 seconds
            setTimeout(() => {
                feedbackMessage.classList.remove('show');
                feedbackButtons.style.display = 'none'; // Keep buttons hidden
            }, 7000);
        } else {
            console.log('Server error:', data);
            // Show error message
            feedbackMessage.className = 'feedback-message error-feedback show';
            feedbackMessage.innerHTML = '⚠️ Error submitting feedback. Please try again.';
            
            // Still hide buttons even on error
            feedbackButtons.style.display = 'none';
        }
    } catch (error) {
        console.error('Feedback error:', error);
        feedbackMessage.className = 'feedback-message error-feedback show';
        feedbackMessage.innerHTML = '⚠️ Error submitting feedback: ' + error.message + '<br><small>Please try again.</small>';
        
        // Still hide buttons on error
        feedbackButtons.style.display = 'none';
    }
}

