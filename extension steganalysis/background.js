// Service worker for context menu, side panel, and drag-and-drop management

// Create context menu on installation
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "sendToSteganalysis",
        title: "Send to Steganalysis Extension",
        contexts: ["image", "video", "audio"]
    });

    // Enable side panel globally
    chrome.sidePanel.setOptions({
        enabled: true
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

            // Open side panel
            chrome.sidePanel.open({ tabId: tab.id });

            // Show notification
            chrome.notifications.create({
                type: 'basic',
                iconUrl: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
                title: 'File Captured',
                message: 'File sent to Steganalysis Extension. Check the side panel.'
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

// Handle messages from content script for drag-and-drop
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'processDraggedFile') {
        (async () => {
            try {
                const { src, filename, tagName, fileType } = message.data;

                // Fetch the media as blob
                const response = await fetch(src);
                const blob = await response.blob();

                // Convert to base64
                const base64 = await blobToBase64(blob);

                // Determine MIME type
                let type = blob.type || 'application/octet-stream';
                if (tagName === 'GALLERY_ITEM' && fileType) {
                    if (fileType === 'image') {
                        type = 'image/png';
                    } else if (fileType === 'video') {
                        type = 'video/mp4';
                    } else if (fileType === 'audio') {
                        type = 'audio/wav';
                    }
                }

                // Store in chrome.storage
                chrome.storage.local.set({
                    'draggedFile': base64,
                    'draggedFileName': filename,
                    'draggedFileType': type
                });

                // Send message to side panel
                chrome.runtime.sendMessage({
                    type: 'draggedFile',
                    base64: base64,
                    filename: filename,
                    type: type
                });

                sendResponse({ success: true });

            } catch (error) {
                console.error('Error processing dragged file:', error);
                sendResponse({ success: false, error: error.message });
            }
        })();
        return true; // Keep the message channel open for async response
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
