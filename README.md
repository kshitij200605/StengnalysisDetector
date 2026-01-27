# Steganalysis Detector

A web-based application for steganography operations including hiding, detecting, and revealing hidden messages in images, videos, and audio files. Includes a Chrome extension for seamless integration with web pages.

## Features

- **Hide Messages**: Embed secret text messages into image, video, and audio files using LSB (Least Significant Bit) steganography
- **Detect Steganography**: Analyze files to detect the presence of hidden messages
- **Reveal Messages**: Extract hidden messages from stego files
- **Gallery**: View and manage stored stego files with database persistence
- **Chrome Extension**: Drag-and-drop functionality directly from web pages
- **API Endpoints**: RESTful API for programmatic access
- **Input Validation**: Secure file handling with comprehensive validation

## Supported File Types

- **Images**: PNG, JPG, JPEG
- **Videos**: MP4, AVI (requires MoviePy)
- **Audio**: WAV, MP3 (requires Pydub)

## Installation

### Prerequisites

- Python 3.7+
- Flask
- PIL (Pillow)
- NumPy
- SQLite3
- Optional: MoviePy (for video processing), Pydub (for audio processing)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd SteganalysisDetector
```

2. Install dependencies:
```bash
pip install flask pillow numpy pydub moviepy
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

### Web Interface

1. **Hide Message**:
   - Upload an image/video/audio file
   - Enter the secret message
   - Click "Hide Message"
   - Download the stego file

2. **Detect Steganography**:
   - Upload a file
   - Click "Run Detection"
   - View detection results

3. **Reveal Message**:
   - Upload a stego file
   - Click "Reveal Message"
   - View the extracted hidden message

### Gallery

- View all stored stego files
- Reveal messages from stored files
- Download files from the database

### API Endpoints

- `POST /api/hide` - Hide message in file
- `POST /api/detect` - Detect steganography
- `POST /api/reveal` - Reveal hidden message

## Chrome Extension

### Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `extension steganalysis` folder

### Features

- Right-click context menu on images/videos
- Drag-and-drop from web pages
- Direct integration with the web app
- Popup interface for quick operations

## Project Structure

```
SteganalysisDetector/
├── app.py                 # Main Flask application
├── extract_stego.py       # Database extraction utility
├── stego.db              # SQLite database
├── static/               # Static files (CSS, JS)
├── templates/            # HTML templates
│   ├── index.html
│   └── gallery.html
├── extension steganalysis/  # Chrome extension
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   ├── popup.css
│   ├── content.js
│   └── background.js
├── Clean/                # Clean sample files
├── Stego/                # Generated stego files
└── README.md
```

## Security Features

- Input validation for filenames and messages
- Path traversal protection
- File type restrictions
- Secure file handling with Werkzeug
- CORS enabled for extension communication

## Technical Details

- **Steganography Method**: LSB substitution
- **Database**: SQLite for file storage
- **Web Framework**: Flask with Jinja2 templates
- **Image Processing**: PIL/Pillow
- **Audio Processing**: Pydub
- **Video Processing**: MoviePy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Disclaimer

This tool is for educational and research purposes. Ensure you have permission to analyze files and comply with applicable laws and regulations.
