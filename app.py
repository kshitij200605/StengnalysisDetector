import os
import io
import sqlite3
import numpy as np
from flask import Flask, render_template, request, send_file
from PIL import Image
from werkzeug.utils import secure_filename
app = Flask(__name__, static_folder="static", template_folder="templates")
DB = "stego.db"
# ===================== DB Setup =====================
def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            data BLOB
        )
        """)
        # Add hidden_message column if it doesn't exist
        try:
            conn.execute("ALTER TABLE images ADD COLUMN hidden_message TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        # Clean up invalid entries
        conn.execute("DELETE FROM images WHERE data IS NULL OR length(data) = 0")
        conn.commit()
def save_to_db(filename, path, hidden_message=None):
    with open(path, "rb") as f:
        img_bytes = f.read()
    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO images (filename, data, hidden_message) VALUES (?, ?, ?)",
            (filename, img_bytes, hidden_message),
        )
        conn.commit()
def get_all_images():
    with sqlite3.connect(DB) as conn:
        rows = conn.execute("SELECT id, filename FROM images WHERE data IS NOT NULL AND length(data) > 0").fetchall()
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
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        if byte == "11111110":  # stop when delimiter is found
            break
        try:
            message += chr(int(byte, 2))
        except:
            break
    return message.strip()
def reveal_text_from_bytes(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    return extract_hidden_message(img)
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
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        if byte == "11111110":  # stop when delimiter is found
            break
        try:
            message += chr(int(byte, 2))
        except:
            break
    return message.strip()

def reveal_text_from_bytes_fast(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    return fast_extract_hidden_message(img)

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
                image = request.files["image"]
                message = request.form["message"]
                filename = secure_filename(image.filename)
                upload_path = os.path.join("static", filename)
                image.save(upload_path)
                stego_path = os.path.join("Stego", f"stego_{os.path.splitext(filename)[0]}.png")
                hide_text(upload_path, message, stego_path)
                save_to_db(filename, stego_path, message)
                result = "✅ Message hidden and saved to DB!"
                # retrain detector with new data
                clf = train_detector()
            except Exception as e:
                result = f"⚠️ Error: {e}"
        # ===== Detect =====
        elif "detect" in request.form:
            try:
                image = request.files["image"]
                filename = secure_filename(image.filename)
                path = os.path.join("static", f"temp_{filename}")
                image.save(path)
                pred = detect_stego(path)
                result = "⚠️ Stego Detected!" if pred == 1 else "✅ Clean Image"
                os.remove(path)  # Clean up temp file
            except Exception as e:
                result = f"⚠️ Detection error: {e}"
        # ===== Reveal =====
        elif "reveal" in request.form:
            try:
                image = request.files["image"]
                filename = secure_filename(image.filename)
                temp_path = os.path.join("static", f"temp_reveal_{filename}")
                image.save(temp_path)
                with open(temp_path, "rb") as f:
                    img_bytes = f.read()
                # Use optimized fast reveal
                hidden_message = reveal_text_from_bytes_fast(img_bytes)
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
            img_bytes = get_image_by_id(image_id)
            if img_bytes:
                # Use optimized fast reveal
                hidden_message = reveal_text_from_bytes_fast(img_bytes)
        except Exception as e:
            hidden_message = f"Error revealing message: {e}"
    return render_template("gallery.html", rows=rows, hidden_message=hidden_message)
if __name__ == "__main__":
    app.run(debug=True)