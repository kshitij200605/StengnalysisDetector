// Content script - simplified to work with context menu
// The drag and drop functionality is handled by the background script via context menus
// to avoid popup closing issues during drag operations

// Optional: Add visual indicator for supported media elements
document.addEventListener('contextmenu', (e) => {
    const target = e.target;
    if (target.tagName && ['IMG', 'VIDEO', 'AUDIO'].includes(target.tagName)) {
        // Could add visual feedback here if needed
    }
});
