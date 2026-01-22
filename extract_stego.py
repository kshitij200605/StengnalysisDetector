import sqlite3

conn = sqlite3.connect('stego.db')
cursor = conn.cursor()

# Get the latest image entry
cursor.execute('SELECT id, filename, data FROM images ORDER BY id DESC LIMIT 1')
row = cursor.fetchone()

if row:
    image_id, filename, data = row
    # Save the image data to a file
    with open(f'stego_{filename}', 'wb') as f:
        f.write(data)
    print(f'Stego image extracted as stego_{filename}')
else:
    print('No images found in database')

conn.close()
