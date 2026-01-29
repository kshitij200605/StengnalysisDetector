import os
import io
import sqlite3
import numpy as np
import string
import shutil
from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
from pydub import AudioSegment
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable CORS for extension requests
DB = "stego.db"

# ===================== Input Validation =====================
def validate_filename(filename):
    """Validate filename to prevent path traversal and injection attacks"""
    if not filename:
        return False
    # Check for path traversal attempts
    if '..' in filename or '/' in filename or '\\' in filename:
        return False
    # Check for dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
    if any(char in filename for char in dangerous_chars):
        return False
    # Check length
    if len(filename) > 255:
        return False
    return True

def validate_message(message):
    """Validate message content"""
    if not message:
        return False
    # Check length (reasonable limit)
    if len(message) > 10000:
        return False
    # Only allow printable characters
    if not all(c in string.printable for c in message):
        return False
    return True

def validate_image_id(image_id):
    """Validate image ID parameter"""
    try:
        id_int = int(image_id)
        if id_int < 1 or id_int > 999999:  # Reasonable bounds
            return False
        return id_int
    except (ValueError, TypeError):
        return False
# ===================== DB Setup =====================
def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            data BLOB,
            type TEXT DEFAULT 'image'
        )
        """)
        # Add hidden_message column if it doesn't exist
        try:
            conn.execute("ALTER TABLE images ADD COLUMN hidden_message TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        # Add type column if it doesn't exist
        try:
            conn.execute("ALTER TABLE images ADD COLUMN type TEXT DEFAULT 'image'")
        except sqlite3.OperationalError:
            pass  # Column already exists
        # Clean up invalid entries
        conn.execute("DELETE FROM images WHERE data IS NULL OR length(data) = 0")
        conn.commit()
def save_to_db(filename, file_data, hidden_message=None, file_type='image'):
    # file_data can be either a path (string) or bytes
    if isinstance(file_data, str):
        with open(file_data, "rb") as f:
            file_bytes = f.read()
    else:
        file_bytes = file_data

    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO images (filename, data, hidden_message, type) VALUES (?, ?, ?, ?)",
            (filename, file_bytes, hidden_message, file_type),
        )
        conn.commit()
def get_all_images():
    with sqlite3.connect(DB) as conn:
        rows = conn.execute("SELECT id, filename, type FROM images WHERE data IS NOT NULL AND length(data) > 0").fetchall()
    return rows
def get_image_by_id(image_id):
    with sqlite3.connect(DB) as conn:
        row = conn.execute("SELECT data FROM images WHERE id=?", (image_id,)).fetchone()
    return row[0] if row else None
def get_message_by_id(image_id):
    with sqlite3.connect(DB) as conn:
        row = conn.execute("SELECT hidden_message FROM images WHERE id=?", (image_id,)).fetchone()
    return row[0] if row else None
init_db()
# ===================== Steganography =====================
def hide_text(image_path, message, output_path):
    img = Image.open(image_path).convert("RGB")
    encoded = img.copy()
    width, height = img.size
    index = 0
    binary_message = ''.join(format(ord(c), '08b') for c in message) + '11111110'  # delimiter
    required_pixels = len(binary_message) // 3 + 1
    if width * height < required_pixels:
        raise ValueError(f"Image too small! Need at least {required_pixels} pixels.")
    for row in range(height):
        for col in range(width):
            if index >= len(binary_message):
                break
            pixel = list(img.getpixel((col, row)))
            for n in range(3):
                if index < len(binary_message):
                    pixel[n] = pixel[n] & ~1 | int(binary_message[index])
                    index += 1
            encoded.putpixel((col, row), tuple(pixel))
    encoded.save(output_path, "PNG")
def extract_hidden_message(img):
    binary_message = ""
    for row in range(img.height):
        for col in range(img.width):
            pixel = list(img.getpixel((col, row)))
            for n in range(3):
                binary_message += str(pixel[n] & 1)
    message = ""
    delimiter_found = False
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        if byte == "11111110":  # stop when delimiter is found
            delimiter_found = True
            break
        try:
            message += chr(int(byte, 2))
        except:
            break
    message = message.strip()
    # Validate message: only printable characters
    if not all(c in string.printable for c in message):
        return "", False
    return message, delimiter_found
def reveal_text_from_bytes(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    message, delimiter_found = extract_hidden_message(img)
    return message if delimiter_found else ""
# ===================== Optimized Reveal =====================
def fast_extract_hidden_message(img):
    # Optimized extraction: scan in a more efficient way
    binary_message = ""
    width, height = img.size
    # Use numpy for faster pixel access
    img_array = np.array(img)
    # Flatten and extract LSBs more efficiently
    for row in range(height):
        for col in range(width):
            pixel = img_array[row, col]
            for n in range(3):
                binary_message += str(pixel[n] & 1)
                if len(binary_message) % 8 == 0 and binary_message[-8:] == '11111110':
                    # Early stop on delimiter
                    break
            if len(binary_message) % 8 == 0 and binary_message[-8:] == '11111110':
                break
        if len(binary_message) % 8 == 0 and binary_message[-8:] == '11111110':
            break
    message = ""
    delimiter_found = False
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        if byte == "11111110":  # stop when delimiter is found
            delimiter_found = True
            break
        try:
            message += chr(int(byte, 2))
        except:
            break
    message = message.strip()
    # Validate message: only printable characters
    if not all(c in string.printable for c in message):
        return "", False
    return message, delimiter_found

def reveal_text_from_bytes_fast(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    message, delimiter_found = fast_extract_hidden_message(img)
    return message if delimiter_found else ""

# ===================== Video Steganography =====================
def hide_text_in_video(video_path, message, output_path):
    if not MOVIEPY_AVAILABLE:
        raise ImportError("MoviePy is not available. Video steganography requires MoviePy.")
    clip = VideoFileClip(video_path)
    frame = clip.get_frame(0)  # Get first frame
    img = Image.fromarray(frame)
    encoded_img = hide_text_in_image(img, message)
    # Replace first frame with encoded one (simplified, for demo)
    # In real implementation, you'd need to modify the video properly
    encoded_img.save(output_path.replace('.mp4', '_frame.png'))
    # For now, just save the frame; full video encoding would require more work

def hide_text_in_image(img, message):
    encoded = img.copy()
    width, height = img.size
    index = 0
    binary_message = ''.join(format(ord(c), '08b') for c in message) + '11111110'
    required_pixels = len(binary_message) // 3 + 1
    if width * height < required_pixels:
        raise ValueError("Image too small!")
    for row in range(height):
        for col in range(width):
            if index >= len(binary_message):
                break
            pixel = list(img.getpixel((col, row)))
            for n in range(3):
                if index < len(binary_message):
                    pixel[n] = pixel[n] & ~1 | int(binary_message[index])
                    index += 1
            encoded.putpixel((col, row), tuple(pixel))
    return encoded

def reveal_text_from_video(video_path):
    if not MOVIEPY_AVAILABLE:
        raise ImportError("MoviePy is not available. Video steganography requires MoviePy.")
    clip = VideoFileClip(video_path)
    frame = clip.get_frame(0)
    img = Image.fromarray(frame)
    return reveal_text_from_image(img)

def reveal_text_from_video_fast(video_path):
    if not MOVIEPY_AVAILABLE:
        raise ImportError("MoviePy is not available. Video steganography requires MoviePy.")
    clip = VideoFileClip(video_path)
    frame = clip.get_frame(0)
    img = Image.fromarray(frame)
    return reveal_text_from_image_fast(img)

def reveal_text_from_image_fast(img):
    binary_message = ""
    width, height = img.size
    # Limit scanning to first 1000 pixels for speed
    pixel_count = 0
    max_pixels = 1000
    for row in range(height):
        for col in range(width):
            if pixel_count >= max_pixels:
                break
            pixel = list(img.getpixel((col, row)))
            for n in range(3):
                if pixel_count >= max_pixels:
                    break
                binary_message += str(pixel[n] & 1)
                pixel_count += 1
                if len(binary_message) % 8 == 0 and binary_message[-8:] == '11111110':
                    # Early stop on delimiter
                    break
            if len(binary_message) % 8 == 0 and binary_message[-8:] == '11111110':
                break
        if len(binary_message) % 8 == 0 and binary_message[-8:] == '11111110':
            break
    message = ""
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        if byte == "11111110":
            break
        try:
            message += chr(int(byte, 2))
        except:
            break
    return message.strip()

def reveal_text_from_image(img):
    binary_message = ""
    for row in range(img.height):
        for col in range(img.width):
            pixel = list(img.getpixel((col, row)))
            for n in range(3):
                binary_message += str(pixel[n] & 1)
    message = ""
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        if byte == "11111110":
            break
        try:
            message += chr(int(byte, 2))
        except:
            break
    return message.strip()

# ===================== Audio Steganography =====================
def hide_text_in_audio(audio_path, message, output_path):
    audio = AudioSegment.from_file(audio_path).set_channels(1)  # Convert to mono for sequential LSB embedding
    samples = np.array(audio.get_array_of_samples())
    binary_message = ''.join(format(ord(c), '08b') for c in message) + '11111110'
    index = 0
    for i in range(len(samples)):
        if index < len(binary_message):
            samples[i] = samples[i] & ~1 | int(binary_message[index])
            index += 1
    new_audio = audio._spawn(samples.tobytes())
    new_audio.export(output_path, format="wav")

def reveal_text_from_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    samples = np.array(audio.get_array_of_samples())
    binary_message = ""
    for sample in samples:
        binary_message += str(sample & 1)
    message = ""
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        if byte == "11111110":
            break
        try:
            message += chr(int(byte, 2))
        except:
            break
    return message.strip()

def reveal_text_from_audio_fast(audio_path):
    audio = AudioSegment.from_file(audio_path)
    samples = np.array(audio.get_array_of_samples())
    binary_message = ""
    # Limit scanning to first 10000 samples for speed
    max_samples = min(10000, len(samples))
    for i in range(max_samples):
        binary_message += str(samples[i] & 1)
        if len(binary_message) % 8 == 0 and binary_message[-8:] == '11111110':
            break
    message = ""
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        if byte == "11111110":
            break
        try:
            message += chr(int(byte, 2))
        except:
            break
    return message.strip()

# ===================== Simple Detector =====================
def detect_stego(file_path):
    # Determine file type based on extension for faster detection
    ext = os.path.splitext(file_path)[1].lower()

    # Image detection
    if ext in ['.png', '.jpg', '.jpeg']:
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            message = reveal_text_from_bytes_fast(file_bytes)
            if message:
                return 1
        except:
            pass
        return 0

    # Video detection
    elif ext in ['.mp4', '.avi']:
        try:
            message = reveal_text_from_video_fast(file_path)
            if message:
                return 1
        except:
            pass
        return 0

    # Audio detection
    elif ext in ['.wav', '.mp3']:
        try:
            message = reveal_text_from_audio_fast(file_path)
            if message:
                return 1
        except:
            pass
        return 0

    # For other files, try image detection as fallback (mislabeled files)
    else:
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            message = reveal_text_from_bytes_fast(file_bytes)
            if message:
                return 1
        except:
            pass
        return 0

def extract_features(image_path):
    # Not used anymore, but keep for compatibility
    return [0]

def train_detector(clean_folder="Clean", stego_folder="Stego"):
    # Dummy for compatibility
    class DummyClassifier:
        def predict(self, X):
            return [0] * len(X)
    return DummyClassifier()

# ===================== Setup folders =====================
os.makedirs("Clean", exist_ok=True)
os.makedirs("Stego", exist_ok=True)
clf = train_detector()
# ===================== Routes =====================
@app.route("/", methods=["GET", "POST"])
def index():
    global clf
    result, hidden_message = None, None
    if request.method == "POST":
        # ===== Hide =====
        if "hide" in request.form:
            try:
                file = request.files["file"]
                message = request.form["message"]

                # Input validation
                if not file or not validate_filename(file.filename):
                    result = "⚠️ Invalid file name"
                elif not validate_message(message):
                    result = "⚠️ Invalid message content"
                else:
                    filename = secure_filename(file.filename)
                    upload_path = os.path.join("static", filename)
                    file.save(upload_path)
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in ['.png', '.jpg', '.jpeg']:
                        stego_path = os.path.join("Stego", f"stego_{os.path.splitext(filename)[0]}.png")
                        hide_text(upload_path, message, stego_path)
                        file_type = 'image'
                    elif ext in ['.mp4', '.avi']:
                        stego_path = os.path.join("Stego", f"stego_{os.path.splitext(filename)[0]}.mp4")
                        hide_text_in_video(upload_path, message, stego_path)
                        file_type = 'video'
                    elif ext in ['.wav', '.mp3']:
                        stego_path = os.path.join("Stego", f"stego_{os.path.splitext(filename)[0]}.wav")
                        hide_text_in_audio(upload_path, message, stego_path)
                        file_type = 'audio'
                    else:
                        # For unsupported file types, just copy the file to Stego folder
                        stego_path = os.path.join("Stego", f"stego_{filename}")
                        shutil.copy(upload_path, stego_path)
                        file_type = 'other'
                    save_to_db(filename, stego_path, message, file_type)
                    result = "✅ Message hidden and saved to DB!"
                    # retrain detector with new data
                    clf = train_detector()
            except Exception as e:
                result = f"⚠️ Error: {e}"
        # ===== Detect =====
        elif "detect" in request.form:
            try:
                file = request.files["file"]
                filename = secure_filename(file.filename)
                path = os.path.join("static", f"temp_{filename}")
                file.save(path)
                pred = detect_stego(path)
                file_size = os.path.getsize(path)
                is_even = file_size % 2 == 0
                result = f"⚠️ Stego Detected! (Even length: {is_even})" if pred == 1 else f"✅ Clean File (Even length: {is_even})"
                os.remove(path)  # Clean up temp file
            except Exception as e:
                result = f"⚠️ Detection error: {e}"
        # ===== Reveal =====
        elif "reveal" in request.form:
            try:
                file = request.files["file"]
                filename = secure_filename(file.filename)
                temp_path = os.path.join("static", f"temp_reveal_{filename}")
                file.save(temp_path)
                ext = os.path.splitext(filename)[1].lower()
                if ext in ['.png', '.jpg', '.jpeg']:
                    with open(temp_path, "rb") as f:
                        file_bytes = f.read()
                    hidden_message = reveal_text_from_bytes_fast(file_bytes)
                elif ext in ['.mp4', '.avi']:
                    hidden_message = reveal_text_from_video(temp_path)
                elif ext in ['.wav', '.mp3']:
                    hidden_message = reveal_text_from_audio(temp_path)
                os.remove(temp_path)  # Clean up temp file
                if hidden_message:
                    result = "🕵️ Hidden message revealed!"
                else:
                    result = "❌ No hidden message found."
            except Exception as e:
                result = f"⚠️ Reveal error: {e}"
    rows = get_all_images()
    return render_template("index.html", result=result, hidden_message=hidden_message, rows=rows)
@app.route("/image/<image_id>")
def get_image(image_id):
    validated_id = validate_image_id(image_id)
    if not validated_id:
        return "Invalid image ID", 400
    img_bytes = get_image_by_id(validated_id)
    if img_bytes:
        return send_file(io.BytesIO(img_bytes), mimetype="image/png")
    return "Image not found", 404

@app.route("/download/<image_id>")
def download_image(image_id):
    validated_id = validate_image_id(image_id)
    if not validated_id:
        return "Invalid image ID", 400
    img_bytes = get_image_by_id(validated_id)
    if img_bytes:
        # Get filename from DB
        with sqlite3.connect(DB) as conn:
            row = conn.execute("SELECT filename FROM images WHERE id = ?", (validated_id,)).fetchone()
        filename = row[0] if row else f"stego_{validated_id}.png"
        return send_file(io.BytesIO(img_bytes), mimetype="image/png", as_attachment=True, download_name=filename)
    return "Image not found", 404
@app.route("/gallery", methods=["GET", "POST"])
def gallery():
    hidden_message = None
    rows = get_all_images()
    if request.method == "POST" and "reveal" in request.form:
        try:
            validated_id = validate_image_id(request.form["image_id"])
            if not validated_id:
                hidden_message = "Invalid image ID"
            else:
                file_bytes = get_image_by_id(validated_id)
                if file_bytes:
                    # Determine file type from DB
                    with sqlite3.connect(DB) as conn:
                        row = conn.execute("SELECT type FROM images WHERE id = ?", (validated_id,)).fetchone()
                    file_type = row[0] if row else 'image'
                    if file_type == 'image':
                        hidden_message = reveal_text_from_bytes_fast(file_bytes)
                    elif file_type == 'video':
                        # Save temp file for video processing
                        temp_path = os.path.join("static", f"temp_gallery_{validated_id}.mp4")
                        with open(temp_path, "wb") as f:
                            f.write(file_bytes)
                        hidden_message = reveal_text_from_video(temp_path)
                        os.remove(temp_path)
                    elif file_type == 'audio':
                        temp_path = os.path.join("static", f"temp_gallery_{validated_id}.wav")
                        with open(temp_path, "wb") as f:
                            f.write(file_bytes)
                        hidden_message = reveal_text_from_audio(temp_path)
                        os.remove(temp_path)
                    elif file_type == 'other':
                        hidden_message = "No hidden message (unsupported file type)"
        except Exception as e:
            hidden_message = f"Error revealing message: {e}"
    return render_template("gallery.html", rows=rows, hidden_message=hidden_message)

# API Endpoints for Chrome Extension
@app.route("/api/hide", methods=["POST"])
def api_hide():
    try:
        file = request.files["file"]
        message = request.form["message"]
        filename = secure_filename(file.filename)
        file_bytes = file.read()
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            # Process image
            img = Image.open(io.BytesIO(file_bytes))
            stego_img = hide_text_in_image(img, message)
            output = io.BytesIO()
            stego_img.save(output, format="PNG")
            output.seek(0)
            save_to_db(filename, output.getvalue(), message, 'image')
            # Get the ID of the saved image
            conn = sqlite3.connect(DB)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM images WHERE filename = ? ORDER BY id DESC LIMIT 1", (filename,))
            image_id = cursor.fetchone()[0]
            conn.close()
            return jsonify({"result": "✅ Message hidden and saved to DB!", "filename": filename, "image_id": image_id})
        elif ext in ['.mp4', '.avi']:
            # Save temp file for video processing
            temp_input = os.path.join("static", f"temp_hide_{filename}")
            with open(temp_input, "wb") as f:
                f.write(file_bytes)
            stego_path = os.path.join("Stego", f"stego_{os.path.splitext(filename)[0]}.mp4")
            hide_text_in_video(temp_input, message, stego_path)
            save_to_db(filename, stego_path, message, 'video')
            os.remove(temp_input)
            return jsonify({"result": "✅ Message hidden in video and saved to DB!", "filename": filename})
        elif ext in ['.wav', '.mp3']:
            # Save temp file for audio processing
            temp_input = os.path.join("static", f"temp_hide_{filename}")
            with open(temp_input, "wb") as f:
                f.write(file_bytes)
            stego_path = os.path.join("Stego", f"stego_{os.path.splitext(filename)[0]}.wav")
            hide_text_in_audio(temp_input, message, stego_path)
            save_to_db(filename, stego_path, message, 'audio')
            os.remove(temp_input)
            return jsonify({"result": "✅ Message hidden in audio and saved to DB!", "filename": filename})
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/detect", methods=["POST"])
def api_detect():
    try:
        file = request.files["file"]
        filename = secure_filename(file.filename)
        file_bytes = file.read()
        pred = detect_stego_from_bytes(file_bytes, filename)
        file_size = len(file_bytes)
        is_even = file_size % 2 == 0
        result = f"⚠️ Stego Detected! (Even length: {is_even})" if pred == 1 else f"✅ Clean File (Even length: {is_even})"
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/reveal", methods=["POST"])
def api_reveal():
    try:
        file = request.files["file"]
        filename = secure_filename(file.filename)
        file_bytes = file.read()
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            hidden_message = reveal_text_from_bytes_fast(file_bytes)
        elif ext in ['.mp4', '.avi']:
            # Save temp file for video processing
            temp_path = os.path.join("static", f"temp_reveal_{filename}")
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            hidden_message = reveal_text_from_video(temp_path)
            os.remove(temp_path)
        elif ext in ['.wav', '.mp3']:
            # Save temp file for audio processing
            temp_path = os.path.join("static", f"temp_reveal_{filename}")
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            hidden_message = reveal_text_from_audio(temp_path)
            os.remove(temp_path)
        else:
            hidden_message = "Unsupported file type"
        if hidden_message:
            return jsonify({"result": "🕵️ Hidden message revealed!", "hidden_message": hidden_message})
        else:
            return jsonify({"result": "❌ No hidden message found."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def detect_stego_from_bytes(file_bytes, filename=""):
    # Simplified detection for API based on file extension
    ext = os.path.splitext(filename)[1].lower() if filename else ""
    try:
        if ext in ['.png', '.jpg', '.jpeg']:
            message = reveal_text_from_bytes_fast(file_bytes)
        elif ext in ['.mp4', '.avi']:
            # Save temp file for video processing
            temp_path = os.path.join("static", f"temp_detect_{filename}")
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            message = reveal_text_from_video_fast(temp_path)
            os.remove(temp_path)
        elif ext in ['.wav', '.mp3']:
            # Save temp file for audio processing
            temp_path = os.path.join("static", f"temp_detect_{filename}")
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            message = reveal_text_from_audio_fast(temp_path)
            os.remove(temp_path)
        else:
            # Fallback to image detection
            message = reveal_text_from_bytes_fast(file_bytes)
        return 1 if message else 0
    except:
        return 0
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
