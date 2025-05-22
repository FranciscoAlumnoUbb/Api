import os
import torch
import pathlib
import requests
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from datetime import datetime
import cv2

# Parche para evitar error de PosixPath en Windows
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
MODEL_PATH = 'model/best.pt'
MODEL_URL = 'https://drive.google.com/uc?export=download&id=11UuwaxJBJA3um5RDX_gnvjoDtn8xHdI_'

# Crear carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# 🔽 Descargar modelo si no existe
def descargar_modelo():
    if not os.path.exists(MODEL_PATH):
        print("🔁 Descargando modelo desde Google Drive...")
        with requests.get(MODEL_URL, stream=True) as r:
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print("✅ Modelo descargado.")

descargar_modelo()

# ✅ Cargar modelo YOLOv5
model = torch.hub.load('yolov5', 'custom', path=MODEL_PATH, source='local')

# 🔍 Endpoint principal
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # Inference
    results = model(image_path)
    results.render()

    # Guardar imagen con bounding boxes
    image_out_path = os.path.join(RESULT_FOLDER, filename)
    img = results.ims[0]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_out_path, img)

    return jsonify({
        'message': 'Prediction complete',
        'result_url': f'/results/{filename}'
    })

# 📸 Servir resultados
@app.route('/results/<filename>', methods=['GET'])
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

# 🚀 Iniciar servidor local
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
