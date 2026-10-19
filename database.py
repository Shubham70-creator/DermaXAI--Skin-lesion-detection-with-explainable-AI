import sqlite3

def init_db():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            predicted_class TEXT,
            confidence REAL,
            iou_score REAL,
            heatmap_path TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(image_path, predicted_class, confidence, iou_score, heatmap_path):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (image_path, predicted_class, confidence, iou_score, heatmap_path)
        VALUES (?, ?, ?, ?, ?)
    ''', (image_path, predicted_class, confidence, iou_score, heatmap_path))
    conn.commit()
    conn.close()

def get_all_predictions():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows