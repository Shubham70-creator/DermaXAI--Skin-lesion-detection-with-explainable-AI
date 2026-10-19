from flask import Flask, render_template, request
from PIL import Image
import os
import csv
import json
from database import init_db, save_prediction, get_all_predictions
from model_utils import load_model, predict_with_explanation

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
HEATMAP_FOLDER = 'static/heatmaps'
PREPROCESSED_FOLDER = 'static/preprocessed'
os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)

init_db()
model = load_model('models/best_model.pth')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filename = file.filename
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    heatmap_filename = 'heatmap_' + filename
    heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)

    predicted_class, confidence, heatmap_path, class_details, top3_list = predict_with_explanation(
        model, image_path, heatmap_path
    )

    confidence_pct = round(confidence * 100, 2)
    low_confidence_warning = confidence_pct < 60

    save_prediction(image_path, predicted_class, confidence, None, heatmap_path)

    return render_template(
        'result.html',
        predicted_class=predicted_class,
        confidence=confidence_pct,
        low_confidence=low_confidence_warning,
        full_name=class_details['full_name'],
        description=class_details['description'],
        risk=class_details['risk'],
        recommendation=class_details['recommendation'],
        original_image=image_path,
        heatmap_image=heatmap_path,
        top3=top3_list
    )
@app.route('/history')
def history():
    records = get_all_predictions()
    return render_template('history.html', records=records)
PERF_FOLDER = os.path.join('static', 'performance')

def read_csv_table(filename, drop=()):
    path = os.path.join(PERF_FOLDER, filename)
    if not os.path.exists(path):
        return None
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    keep = [i for i, h in enumerate(rows[0]) if h not in drop]
    return [[r[i] for i in keep] for r in rows]

def read_summary():
    path = os.path.join(PERF_FOLDER, 'research_results.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f).get('densenet169_final')

@app.route('/performance')
def performance():
    return render_template(
        'performance.html',
        summary=read_summary(),
        model_table=read_csv_table('table_model_comparison.csv', drop=('Epoch time (s)',)),
        ablation_table=read_csv_table('table_ablation.csv'),
        xai_table=read_csv_table('table_xai_comparison.csv'),
    )
if __name__ == '__main__':
    app.run(debug=True)