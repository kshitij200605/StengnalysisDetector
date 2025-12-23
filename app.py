import os
import io
import sqlite3
import numpy as np
import string
from flask import Flask, render_template, request, send_file
from PIL import Image
from werkzeug.utils import secure_filename
try:   
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
from pydub import AudioSegment
app = Flask(__name__, static_folder="static", template_folder="templates")
DB = "stego.db"
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
def save_to_db(filename, path, hidden_message=None, file_type='image'):
    with open(path, "rb") as f:
        file_bytes = f.read()
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
    audio = AudioSegment.from_file(audio_path)
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

# ===================== Simple Detector =====================
def detect_stego(image_path):
    # Accurate detection: try to extract hidden message
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    message = reveal_text_from_bytes_fast(img_bytes)
    return 1 if message else 0  # stego if message found

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
                    raise ValueError("Unsupported file type")
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
@app.route("/image/<int:image_id>")
def get_image(image_id):
    img_bytes = get_image_by_id(image_id)
    if img_bytes:
        return send_file(io.BytesIO(img_bytes), mimetype="image/png")
    return "Image not found", 404
@app.route("/gallery", methods=["GET", "POST"])
def gallery():
    hidden_message = None
    rows = get_all_images()
    if request.method == "POST" and "reveal" in request.form:
        try:
            image_id = int(request.form["image_id"])
            file_bytes = get_image_by_id(image_id)
            if file_bytes:
                # Determine file type from DB
                with sqlite3.connect(DB) as conn:
                    row = conn.execute("SELECT type FROM images WHERE id=?", (image_id,)).fetchone()
                file_type = row[0] if row else 'image'
                if file_type == 'image':
                    hidden_message = reveal_text_from_bytes_fast(file_bytes)
                elif file_type == 'video':
                    # Save temp file for video processing
                    temp_path = os.path.join("static", f"temp_gallery_{image_id}.mp4")
                    with open(temp_path, "wb") as f:
                        f.write(file_bytes)
                    hidden_message = reveal_text_from_video(temp_path)
                    os.remove(temp_path)
                elif file_type == 'audio':
                    temp_path = os.path.join("static", f"temp_gallery_{image_id}.wav")
                    with open(temp_path, "wb") as f:
                        f.write(file_bytes)
                    hidden_message = reveal_text_from_audio(temp_path)
                    os.remove(temp_path)
        except Exception as e:
            hidden_message = f"Error revealing message: {e}"
    return render_template("gallery.html", rows=rows, hidden_message=hidden_message)
if __name__ == "__main__":
    app.run(debug=True)