import os
import torch
import pathlib
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from datetime import datetime

# Parche para evitar error de PosixPath en Windows
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
MODEL_PATH = 'model/best.pt'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Cargar modelo YOLOv5
model = torch.hub.load('yolov5', 'custom', path=MODEL_PATH, source='local')

# Endpoint principal
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # Realizar inferencia
    results = model(image_path)

    # Dibujar directamente sobre la imagen
    results.render()

    # Guardar resultado sobrescribiendo
    import cv2
    image_out_path = os.path.join(RESULT_FOLDER, filename)
    img = results.ims[0]  # Imagen con las detecciones
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_out_path, img)

    return jsonify({
        'message': 'Prediction complete',
        'result_url': f'/results/{filename}'
    })

# Servir imágenes procesadas
@app.route('/results/<filename>', methods=['GET'])
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

# 🚀 Esta línea inicia el servidor Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
