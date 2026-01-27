// Global variable to store dragged file
let draggedFile = null;

// Function to get dragged file from chrome.storage
async function getDraggedFile() {
    return new Promise((resolve, reject) => {
        chrome.storage.local.get(['draggedFile', 'draggedFileName', 'draggedFileType'], (result) => {
            if (chrome.runtime.lastError) {
                reject(new Error('Failed to access storage'));
                return;
            }

            if (result.draggedFile) {
                // Convert base64 back to File object
                const base64 = result.draggedFile;
                const name = result.draggedFileName || 'dragged_file';
                const type = result.draggedFileType || 'application/octet-stream';

                fetch(base64)
                    .then(res => res.blob())
                    .then(blob => {
                        const file = new File([blob], name, { type: type });
                        resolve(file);
                    })
                    .catch(reject);
            } else {
                reject(new Error('No dragged file available'));
            }
        });
    });
}

// Function to clear dragged file from storage
function clearDraggedFile() {
    chrome.storage.local.remove(['draggedFile', 'draggedFileName', 'draggedFileType']);
}

// Show dragged file name
function showDraggedFile(filename) {
    document.getElementById('dragged-file-name').textContent = filename;
    document.getElementById('dragged-file-display').style.display = 'block';
}

// Show result message
function showResult(message, type = 'success') {
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = message;
    resultDiv.className = `result ${type}`;
    resultDiv.style.display = 'block';
}

// Hide result message
function hideResult() {
    document.getElementById('result').style.display = 'none';
}

// Show hidden message
function showHiddenMessage(message) {
    const messageDiv = document.getElementById('hidden-message');
    document.getElementById('message-content').textContent = message;
    messageDiv.style.display = 'block';
}

// Handle hide message
async function handleHide(e) {
    e.preventDefault();
    const message = document.getElementById('hide-message').value;
    if (!message) {
        showResult('Please enter a message to hide', 'error');
        return;
    }

    let file = draggedFile || document.getElementById('hide-file').files[0];

    if (!file) {
        showResult('Please select a file or drag one from a web page', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('message', message);

    try {
        const response = await fetch('http://localhost:5000/api/hide', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        if (response.ok) {
            showResult(result.result, 'success');
        } else {
            showResult(result.error || 'Error hiding message', 'error');
        }
    } catch (error) {
        showResult('Failed to connect to server. Make sure Flask app is running on port 5000.', 'error');
    }
}

// Handle detect steganography
async function handleDetect() {
    let file = draggedFile || document.getElementById('detect-file').files[0];

    // Try to get dragged file from content script if none selected
    if (!file) {
        try {
            file = await getDraggedFile();
        } catch (error) {
            showResult('Please drag a file from a web page or select one', 'error');
            return;
        }
    }

    if (!file) {
        showResult('Please drag a file from a web page or select one', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:5000/api/detect', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        if (response.ok) {
            showResult(result.result, 'success');
        } else {
            showResult(result.error || 'Error detecting steganography', 'error');
        }
    } catch (error) {
        showResult('Failed to connect to server. Make sure Flask app is running on port 5000.', 'error');
    }
}

// Handle reveal message
async function handleReveal(e) {
    e.preventDefault();
    let file = draggedFile || document.getElementById('reveal-file').files[0];

    // Try to get dragged file from content script if none selected
    if (!file) {
        try {
            file = await getDraggedFile();
        } catch (error) {
            showResult('Please drag a file from a web page or select one', 'error');
            return;
        }
    }

    if (!file) {
        showResult('Please drag a file from a web page or select one', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:5000/api/reveal', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        if (response.ok) {
            showResult(result.result, 'success');
            if (result.hidden_message) {
                showHiddenMessage(result.hidden_message);
            }
        } else {
            showResult(result.error || 'Error revealing message', 'error');
        }
    } catch (error) {
        showResult('Failed to connect to server. Make sure Flask app is running on port 5000.', 'error');
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Set up event listeners
    document.getElementById('hide-btn').addEventListener('click', handleHide);
    document.getElementById('detect-btn').addEventListener('click', handleDetect);
    document.getElementById('reveal-btn').addEventListener('click', handleReveal);
    document.getElementById('gallery-btn').addEventListener('click', () => {
        chrome.tabs.create({ url: 'http://localhost:5000/gallery' });
    });

    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => {
            // Remove active from all tabs
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            // Add active to clicked
            button.classList.add('active');
            const tabId = button.getAttribute('data-tab') + '-tab';
            document.getElementById(tabId).classList.add('active');
        });
    });

    // File input change listeners
    document.getElementById('hide-file').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            document.getElementById('hide-file-name').textContent = e.target.files[0].name;
        }
    });

    document.getElementById('detect-file').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            document.getElementById('detect-file-name').textContent = e.target.files[0].name;
        }
    });

    document.getElementById('reveal-file').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            document.getElementById('reveal-file-name').textContent = e.target.files[0].name;
        }
    });

    // Drag and drop event listeners for direct drops
    const container = document.querySelector('.container');
    container.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        container.classList.add('drag-over');
    });

    container.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        container.classList.remove('drag-over');
    });

    container.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        container.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            draggedFile = file;
            showDraggedFile(file.name);

            // Determine active tab and set file accordingly
            const activeTab = document.querySelector('.tab-button.active').getAttribute('data-tab');
            if (activeTab === 'hide') {
                document.getElementById('hide-file').files = files;
                document.getElementById('hide-file-name').textContent = file.name;
            } else if (activeTab === 'detect') {
                document.getElementById('detect-file').files = files;
                document.getElementById('detect-file-name').textContent = file.name;
            } else if (activeTab === 'reveal') {
                document.getElementById('reveal-file').files = files;
                document.getElementById('reveal-file-name').textContent = file.name;
            }
        }
    });

    // Listen for messages from content script
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (message.type === 'draggedFile') {
            // Handle dragged file from content script
            const { base64, filename, type } = message;
            fetch(base64)
                .then(res => res.blob())
                .then(blob => {
                    draggedFile = new File([blob], filename, { type: type });
                    showDraggedFile(filename);
                })
                .catch(error => console.error('Error processing dragged file:', error));
        }
    });
});
