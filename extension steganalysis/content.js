// Content script for drag-and-drop functionality

let draggedElement = null;

document.addEventListener('dragstart', (e) => {
    const target = e.target;
    if (target.tagName === 'IMG' || target.tagName === 'VIDEO' || target.tagName === 'AUDIO' ||
        (target.tagName === 'A' && (target.href.match(/\.(jpg|jpeg|png|gif|mp4|avi|mp3|wav)$/i))) ||
        target.classList.contains('thumb-card')) {
        draggedElement = target;
        // Allow drag
    }
});

document.addEventListener('dragend', (e) => {
    if (draggedElement) {
        const target = e.target;
        if (target.tagName === 'IMG' || target.tagName === 'VIDEO' || target.tagName === 'AUDIO' ||
            (target.tagName === 'A' && (target.href.match(/\.(jpg|jpeg|png|gif|mp4|avi|mp3|wav)$/i)))) {
            // Get the source URL
            let src = '';
            let filename = '';
            if (target.tagName === 'IMG') {
                src = target.src;
                filename = src.split('/').pop() || 'image.jpg';
            } else if (target.tagName === 'VIDEO') {
                src = target.src || target.currentSrc;
                filename = src.split('/').pop() || 'video.mp4';
            } else if (target.tagName === 'AUDIO') {
                src = target.src || target.currentSrc;
                filename = src.split('/').pop() || 'audio.mp3';
            } else if (target.tagName === 'A') {
                src = target.href;
                filename = src.split('/').pop() || 'file';
            }

            if (src) {
                // Send message to background script
                chrome.runtime.sendMessage({
                    type: 'processDraggedFile',
                    data: {
                        src: src,
                        filename: filename,
                        tagName: target.tagName
                    }
                });
            }
        } else if (target.classList.contains('thumb-card')) {
            // Handle gallery thumb-card drag
            const imageId = target.getAttribute('data-id');
            const filename = target.getAttribute('data-filename');
            const fileType = target.getAttribute('data-type');
            if (imageId && filename) {
                // Fetch the file from the server
                const src = `http://localhost:5000/image/${imageId}`;
                chrome.runtime.sendMessage({
                    type: 'processDraggedFile',
                    data: {
                        src: src,
                        filename: filename,
                        tagName: 'GALLERY_ITEM',
                        fileType: fileType
                    }
                });
            }
        }
        draggedElement = null;
    }
});

// Optional: Add visual indicator for supported media elements
document.addEventListener('contextmenu', (e) => {
    const target = e.target;
    if (target.tagName && ['IMG', 'VIDEO', 'AUDIO'].includes(target.tagName)) {
        // Could add visual feedback here if needed
    }
});
