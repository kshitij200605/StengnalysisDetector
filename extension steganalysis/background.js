// Service worker for context menu and popup management

// Create context menu on installation
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "sendToSteganalysis",
        title: "Send to Steganalysis Extension",
        contexts: ["image", "video", "audio"]
    });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId === "sendToSteganalysis") {
        try {
            // Get the media URL
            const mediaUrl = info.srcUrl;
            if (!mediaUrl) return;

            // Fetch the media as blob
            const response = await fetch(mediaUrl);
            const blob = await response.blob();

            // Convert to base64
            const base64 = await blobToBase64(blob);

            // Store in chrome.storage
            const filename = mediaUrl.split('/').pop() || 'media_file';
            const type = blob.type || 'application/octet-stream';

            chrome.storage.local.set({
                'draggedFile': base64,
                'draggedFileName': filename,
                'draggedFileType': type
            });

            // Show notification
            chrome.notifications.create({
                type: 'basic',
                iconUrl: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
                title: 'File Captured',
                message: 'File sent to Steganalysis Extension. Open the extension to use it.'
            });

        } catch (error) {
            console.error('Error processing media:', error);
            chrome.notifications.create({
                type: 'basic',
                iconUrl: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
                title: 'Error',
                message: 'Failed to capture file.'
            });
        }
    }
});

// Helper function to convert blob to base64
function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}
