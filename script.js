// --- Element Selectors ---
const predictionForm = document.getElementById('prediction-form');
const resultContainer = document.getElementById('result-container');
const predictionText = document.getElementById('prediction-text');
const confidenceText = document.getElementById('confidence-text');
const classifyBtn = document.getElementById('classify-btn');

const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatSendBtn = document.getElementById('chat-send-btn');
const chatWindow = document.getElementById('chat-window');
const chatContainer = document.getElementById('chat-container');
const chatToggle = document.getElementById('chat-toggle');
const chatClose = document.getElementById('chat-close');

// --- API URLs ---
// Automatically detect environment
const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://localhost:5000' 
    : window.location.origin; 

// --- State Management ---
let predictionContext = null; // This will store the data for the chatbot
let selectedFile = null; // Store the selected file for batch analysis
let batchResults = null; // Store batch analysis results

// --- Feature metadata & sliders ---
let featureMetadata = null;

async function loadFeatureMetadata() {
    try {
        const res = await fetch(`${API_URL}/feature-metadata`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        featureMetadata = await res.json();
        applyRangesToSliders();
        setupBatchGlossary();
    } catch (e) {
        console.warn('Failed to load feature metadata, using defaults', e);
        applyRangesToSliders();
    }
}

function setRangeAttrs(id, fallbackMin, fallbackMax) {
    const input = document.getElementById(id);
    if (!input) return;
    const rv = document.querySelector(`.range-value[data-for="${id}"]`);
    const ranges = (featureMetadata && featureMetadata.ranges && featureMetadata.ranges[id]) || null;
    const min = ranges && ranges.min != null ? ranges.min : fallbackMin;
    const max = ranges && ranges.max != null ? ranges.max : fallbackMax;
    if (min != null) input.min = min;
    if (max != null) input.max = max;
    if (rv) rv.textContent = input.value;
    input.addEventListener('input', () => { if (rv) rv.textContent = input.value; });
}

function applyRangesToSliders() {
    setRangeAttrs('koi_score', 0, 1);
    setRangeAttrs('koi_period', null, null);
    setRangeAttrs('koi_duration', null, null);
    setRangeAttrs('koi_depth', null, null);
    setRangeAttrs('koi_prad', null, null);
}

function setupBatchGlossary() {
    // Populate tooltips/descriptions in required columns list
    try {
        if (!featureMetadata || !featureMetadata.descriptions) return;
        const container = document.getElementById('required-columns');
        if (!container) return;
        Array.from(container.children).forEach(span => {
            const key = span.textContent;
            const desc = featureMetadata.descriptions[key];
            if (desc) {
                span.title = desc;
            }
        });
    } catch(_) {}
}

// Load metadata on DOM ready
document.addEventListener('DOMContentLoaded', loadFeatureMetadata);

// --- Prediction Form Logic ---
predictionForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    classifyBtn.textContent = 'Analyzing...';
    classifyBtn.disabled = true;

    // Collect data from all input fields
    const formData = {
        'koi_score': parseFloat(document.getElementById('koi_score').value),
        'koi_fpflag_nt': parseFloat(document.getElementById('koi_fpflag_nt').value),
        'koi_fpflag_ss': parseFloat(document.getElementById('koi_fpflag_ss').value),
        'koi_fpflag_co': parseFloat(document.getElementById('koi_fpflag_co').value),
        'koi_fpflag_ec': parseFloat(document.getElementById('koi_fpflag_ec').value),
        'koi_period': parseFloat(document.getElementById('koi_period').value),
        'koi_period_err1': parseFloat(document.getElementById('koi_period_err1').value),
        'koi_period_err2': parseFloat(document.getElementById('koi_period_err2').value),
        'koi_time0bk': parseFloat(document.getElementById('koi_time0bk').value),
        'koi_time0bk_err1': parseFloat(document.getElementById('koi_time0bk_err1').value),
        'koi_time0bk_err2': parseFloat(document.getElementById('koi_time0bk_err2').value),
        'koi_impact': parseFloat(document.getElementById('koi_impact').value),
        'koi_impact_err1': parseFloat(document.getElementById('koi_impact_err1').value),
        'koi_impact_err2': parseFloat(document.getElementById('koi_impact_err2').value),
        'koi_duration': parseFloat(document.getElementById('koi_duration').value),
        'koi_duration_err1': parseFloat(document.getElementById('koi_duration_err1').value),
        'koi_duration_err2': parseFloat(document.getElementById('koi_duration_err2').value),
        'koi_depth': parseFloat(document.getElementById('koi_depth').value),
        'koi_depth_err1': parseFloat(document.getElementById('koi_depth_err1').value),
        'koi_depth_err2': parseFloat(document.getElementById('koi_depth_err2').value),
        'koi_prad': parseFloat(document.getElementById('koi_prad').value),
        'koi_prad_err1': parseFloat(document.getElementById('koi_prad_err1').value),
        'koi_prad_err2': parseFloat(document.getElementById('koi_prad_err2').value),
        'koi_teq': parseFloat(document.getElementById('koi_teq').value),
        'koi_teq_err1': parseFloat(document.getElementById('koi_teq_err1').value),
        'koi_teq_err2': parseFloat(document.getElementById('koi_teq_err2').value),
        'koi_insol': parseFloat(document.getElementById('koi_insol').value),
        'koi_insol_err1': parseFloat(document.getElementById('koi_insol_err1').value),
        'koi_insol_err2': parseFloat(document.getElementById('koi_insol_err2').value),
        'koi_model_snr': parseFloat(document.getElementById('koi_model_snr').value),
        'koi_tce_plnt_num': parseFloat(document.getElementById('koi_tce_plnt_num').value),
        'koi_steff': parseFloat(document.getElementById('koi_steff').value),
        'koi_steff_err1': parseFloat(document.getElementById('koi_steff_err1').value),
        'koi_steff_err2': parseFloat(document.getElementById('koi_steff_err2').value),
        'koi_slogg': parseFloat(document.getElementById('koi_slogg').value),
        'koi_slogg_err1': parseFloat(document.getElementById('koi_slogg_err1').value),
        'koi_slogg_err2': parseFloat(document.getElementById('koi_slogg_err2').value),
        'koi_srad': parseFloat(document.getElementById('koi_srad').value),
        'koi_srad_err1': parseFloat(document.getElementById('koi_srad_err1').value),
        'koi_srad_err2': parseFloat(document.getElementById('koi_srad_err2').value),
        'ra': parseFloat(document.getElementById('ra').value),
        'dec': parseFloat(document.getElementById('dec').value),
        'koi_kepmag': parseFloat(document.getElementById('koi_kepmag').value)
    };

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData),
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const result = await response.json();

        // Display result
        predictionText.textContent = `Prediction: ${result.prediction}`;
        confidenceText.textContent = `Confidence: ${result.confidence}`;
        resultContainer.classList.remove('hidden');

        // *** IMPORTANT: Store context for the chatbot ***
        predictionContext = {
            input_data: formData,
            prediction: result.prediction,
            confidence: result.confidence
        };

        // Enable the chat
        chatInput.disabled = false;
        chatSendBtn.disabled = false;
        addBotMessage("I've analyzed the data! What would you like to know about this candidate?");

    } catch (error) {
        console.error('Error:', error);
        predictionText.textContent = 'Error making prediction.';
        confidenceText.textContent = '';
    } finally {
        classifyBtn.textContent = 'CLASSIFY CANDIDATE';
        classifyBtn.disabled = false;
    }
});

// --- Chat Toggle Functionality ---
chatToggle.addEventListener('click', () => {
    chatContainer.classList.toggle('open');
});

chatClose.addEventListener('click', () => {
    chatContainer.classList.remove('open');
});

// --- Chat Form Logic ---
chatForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const userMessage = chatInput.value.trim();
    if (!userMessage) return;

    // Display user message and clear input
    addUserMessage(userMessage);
    chatInput.value = '';

    if (!predictionContext) {
        addBotMessage("Please classify a candidate first before asking questions.");
        return;
    }
    
    // Show loading indicator
    const loadingIndicator = addLoadingMessage();

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: userMessage,
                context: predictionContext
            }),
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const result = await response.json();

        // Remove loading indicator and add bot's reply
        loadingIndicator.remove();
        addBotMessage(result.reply);

    } catch (error) {
        console.error('Chat Error:', error);
        loadingIndicator.remove();
        addBotMessage("Sorry, I'm having trouble connecting to my knowledge base. Please try again.");
    }
});

// --- Chat Helper Functions ---
function addUserMessage(message) {
    const messageElement = document.createElement('div');
    messageElement.className = 'user-message';
    messageElement.innerHTML = `<p>${message}</p>`;
    chatWindow.appendChild(messageElement);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addBotMessage(message) {
    const messageElement = document.createElement('div');
    messageElement.className = 'bot-message';
    messageElement.innerHTML = `<p>${message}</p>`; // Use innerHTML to render formatting from Gemini if any
    chatWindow.appendChild(messageElement);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addLoadingMessage() {
    const loadingElement = document.createElement('div');
    loadingElement.className = 'bot-message loading-message';
    loadingElement.innerHTML = `<p><i>Exo-Chat is thinking...</i></p>`;
    chatWindow.appendChild(loadingElement);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return loadingElement;
}

// --- Batch Analysis Functionality ---

// Mode switching
const modeBtns = document.querySelectorAll('.mode-btn');
const singleMode = document.getElementById('single-mode');
const batchMode = document.getElementById('batch-mode');

modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        
        // Update button states
        modeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Show/hide modes
        if (mode === 'single') {
            singleMode.classList.remove('hidden');
            batchMode.classList.add('hidden');
        } else {
            singleMode.classList.add('hidden');
            batchMode.classList.remove('hidden');
        }
    });
});

// File upload functionality
const uploadBox = document.getElementById('upload-box');
const fileInput = document.getElementById('csv-file-input');
const browseBtn = document.getElementById('browse-btn');
const fileInfo = document.getElementById('file-info');
const uploadBtn = document.getElementById('upload-btn');
const removeFileBtn = document.getElementById('remove-file');

// Browse button
browseBtn.addEventListener('click', () => fileInput.click());

// File input change
fileInput.addEventListener('change', handleFileSelect);

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Click to upload
uploadBox.addEventListener('click', () => fileInput.click());

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    selectedFile = file;
    
    // Show file info
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = formatFileSize(file.size);
    fileInfo.classList.remove('hidden');
    uploadBtn.classList.remove('hidden');
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Remove file
removeFileBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.classList.add('hidden');
    uploadBtn.classList.add('hidden');
});

// Upload and analyze
uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    uploadBtn.textContent = 'Analyzing...';
    uploadBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch(`${API_URL}/upload-csv`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            batchResults = result;
            displayBatchResults(result);
        } else {
            alert(`Error: ${result.error}`);
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Upload failed. Please try again.');
    } finally {
        uploadBtn.textContent = 'ANALYZE BATCH';
        uploadBtn.disabled = false;
    }
});

function displayBatchResults(results) {
    const resultsDiv = document.getElementById('batch-results');
    const summaryDiv = document.getElementById('results-summary');
    const tbody = document.getElementById('results-tbody');
    
    // Show results
    resultsDiv.classList.remove('hidden');
    
    // Update summary
    summaryDiv.innerHTML = `
        <div class="summary-item">
            <div class="number">${results.total_processed}</div>
            <div class="label">Total Processed</div>
        </div>
        <div class="summary-item">
            <div class="number">${results.successful}</div>
            <div class="label">Successful</div>
        </div>
        <div class="summary-item">
            <div class="number">${results.total_processed - results.successful}</div>
            <div class="label">Errors</div>
        </div>
    `;
    
    // Update table
    tbody.innerHTML = '';
    results.results.forEach(result => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${result.id}</td>
            <td class="${getPredictionClass(result.prediction)}">${result.prediction}</td>
            <td>${result.confidence}</td>
            <td class="${result.status === 'success' ? 'success' : 'error'}">${result.status}</td>
        `;
        tbody.appendChild(row);
    });
}

function getPredictionClass(prediction) {
    switch(prediction) {
        case 'CONFIRMED': return 'confirmed';
        case 'CANDIDATE': return 'candidate';
        case 'FALSE POSITIVE': return 'false-positive';
        default: return '';
    }
}

// Download template
document.getElementById('download-template').addEventListener('click', () => {
    const templateData = {
        'koi_score': 0.5,
        'koi_fpflag_nt': 0,
        'koi_fpflag_ss': 0,
        'koi_fpflag_co': 0,
        'koi_fpflag_ec': 0,
        'koi_period': 9.12,
        'koi_period_err1': 0.001,
        'koi_period_err2': -0.001,
        'koi_time0bk': 1234.5,
        'koi_time0bk_err1': 0.01,
        'koi_time0bk_err2': -0.01,
        'koi_impact': 0.5,
        'koi_impact_err1': 0.1,
        'koi_impact_err2': -0.1,
        'koi_duration': 3.5,
        'koi_duration_err1': 0.1,
        'koi_duration_err2': -0.1,
        'koi_depth': 150.0,
        'koi_depth_err1': 10.0,
        'koi_depth_err2': -10.0,
        'koi_prad': 1.5,
        'koi_prad_err1': 0.1,
        'koi_prad_err2': -0.1,
        'koi_teq': 300.0,
        'koi_teq_err1': 10.0,
        'koi_teq_err2': -10.0,
        'koi_insol': 1.0,
        'koi_insol_err1': 0.1,
        'koi_insol_err2': -0.1,
        'koi_model_snr': 10.0,
        'koi_tce_plnt_num': 1,
        'koi_steff': 5778.0,
        'koi_steff_err1': 50.0,
        'koi_steff_err2': -50.0,
        'koi_slogg': 4.44,
        'koi_slogg_err1': 0.1,
        'koi_slogg_err2': -0.1,
        'koi_srad': 1.0,
        'koi_srad_err1': 0.1,
        'koi_srad_err2': -0.1,
        'ra': 180.0,
        'dec': 30.0,
        'koi_kepmag': 12.0
    };
    
    const headers = Object.keys(templateData);
    const values = Object.values(templateData);
    const csv = [headers.join(','), values.join(',')].join('\n');
    downloadCSV(csv, 'exoplanet_template.csv');
});

// Download results
document.getElementById('download-results').addEventListener('click', () => {
    if (!batchResults) return;
    
    const csv = generateCSV(batchResults.results);
    downloadCSV(csv, 'exoplanet_predictions.csv');
});

function generateCSV(results) {
    const headers = ['ID', 'Prediction', 'Confidence', 'Status'];
    const rows = results.map(r => [r.id, r.prediction, r.confidence, r.status]);
    return [headers, ...rows].map(row => row.join(',')).join('\n');
}

function downloadCSV(csv, filename) {
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
}

// Quick test button
document.getElementById('quick-test-btn').addEventListener('click', () => {
    // Fill form with sample data
    document.getElementById('koi_score').value = 0.8;
    document.getElementById('koi_period').value = 12.5;
    document.getElementById('koi_duration').value = 4.2;
    document.getElementById('koi_depth').value = 200.0;
    document.getElementById('koi_prad').value = 2.1;
});

// Collapsible sections
document.querySelectorAll('.section-header').forEach(header => {
    header.addEventListener('click', () => {
        const section = header.parentElement;
        section.classList.toggle('collapsed');
    });
});

// Populate required columns on page load
document.addEventListener('DOMContentLoaded', () => {
    const requiredColumns = [
        'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
        'koi_period', 'koi_period_err1', 'koi_period_err2', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2',
        'koi_impact', 'koi_impact_err1', 'koi_impact_err2', 'koi_duration', 'koi_duration_err1', 'koi_duration_err2',
        'koi_depth', 'koi_depth_err1', 'koi_depth_err2', 'koi_prad', 'koi_prad_err1', 'koi_prad_err2',
        'koi_teq', 'koi_teq_err1', 'koi_teq_err2', 'koi_insol', 'koi_insol_err1', 'koi_insol_err2',
        'koi_model_snr', 'koi_tce_plnt_num', 'koi_steff', 'koi_steff_err1', 'koi_steff_err2',
        'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2', 'koi_srad', 'koi_srad_err1', 'koi_srad_err2',
        'ra', 'dec', 'koi_kepmag'
    ];

    const columnsContainer = document.getElementById('required-columns');
    requiredColumns.forEach(column => {
        const span = document.createElement('span');
        span.textContent = column;
        columnsContainer.appendChild(span);
    });

    // Create scattered stars
    createStars('.stars', 200, 1, 0.8);
    createStars('.stars2', 100, 2, 0.5);
    createStars('.stars3', 50, 3, 0.3);
});

// Function to create randomly scattered stars
function createStars(containerSelector, count, size, opacity) {
    const container = document.querySelector(containerSelector);
    for (let i = 0; i < count; i++) {
        const star = document.createElement('div');
        star.style.position = 'absolute';
        star.style.width = size + 'px';
        star.style.height = size + 'px';
        star.style.backgroundColor = 'white';
        star.style.borderRadius = '50%';
        star.style.opacity = opacity;
        star.style.left = Math.random() * 100 + '%';
        star.style.top = Math.random() * 100 + '%';
        star.style.animation = `twinkle ${2 + Math.random() * 3}s infinite alternate`;
        container.appendChild(star);
    }
}