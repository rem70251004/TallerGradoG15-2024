from flask import Flask, render_template, jsonify, Response, request
from flask_cors import CORS
import base64
import numpy as np
import cv2
import os
import threading 
from create_dataset import DataCollector
from train_classifier import ModelTrainer 
from inference_classifier import ModelHandler
from LSB.normalize_samples import NORMALIZE_CLASS
from LSB.create_keypoints import KeypointCreator
from LSB.training_model import TRAINING_MODEL


app = Flask(__name__)
CORS(app)
model_handler = ModelHandler() 
model_trainer = ModelTrainer()
normalize_class = NORMALIZE_CLASS()
keypoint_creator = KeypointCreator()
training_model = TRAINING_MODEL()

DATA_DIR_ABECEDARIO = './data_abecedario'
DATA_DIR_PALABRAS = './data_palabras'
# Asegúrate de que los directorios existen
os.makedirs(DATA_DIR_ABECEDARIO, exist_ok=True)
os.makedirs(DATA_DIR_PALABRAS, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<sw>')
def video_feed(sw):
    if sw == '1':  # Asegúrate de que 'sw' se evalúe correctamente
        return Response(model_handler.generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif sw == '0':
        return Response(model_handler.run_abecedario_model(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif sw == '2':
        return Response(model_handler.run_palabras_model(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else :
        return Response(model_handler.capture_samples(sw),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


def decode_image(image_data):
    """Decodifica una imagen en formato base64 y devuelve un frame de OpenCV."""
    header, encoded = image_data.split(',', 1)  # Dividir el encabezado de los datos
    image_bytes = base64.b64decode(encoded)  # Decodificar Base64 a bytes
    np_array = np.frombuffer(image_bytes, np.uint8)  # Convertir bytes a un array NumPy
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Decodificar el array NumPy a una imagen

    if frame is None or frame.size == 0:
        raise ValueError("La imagen está vacía o no se pudo decodificar correctamente.")
    
    return frame

@app.route('/collect-images/abecedario/<class_name>', methods=['POST'])
def collect_abecedario_images(class_name):
    data = request.get_json()
    image_data = data.get('image') 

    if not image_data:
        return jsonify({"message": "No image data provided."}), 400

    try:
        frame = decode_image(image_data)

        # Crear un directorio para la clase si no existe
        class_dir = os.path.join(DATA_DIR_ABECEDARIO, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Generar un nombre de archivo basado en la cantidad de imágenes en la carpeta de la clase
        image_count = len(os.listdir(class_dir))
        image_name = f"{image_count + 1}.jpg"
        image_path = os.path.join(class_dir, image_name)

        # Guardar la imagen en el directorio de la clase
        cv2.imwrite(image_path, frame)

        return jsonify({"message": f"Abecedario img collected for class {class_name}", "image_path": image_path}), 200
            
    except Exception as e:
        return jsonify({"message": "Error al recolectar imágenes del abecedario: " + str(e)}), 500

@app.route('/collect-images/palabras/<class_name>', methods=['POST'])
def collect_palabras_images(class_name):
    data = request.get_json()
    image_data = data.get('image') 

    if not image_data:
        return jsonify({"message": "No image data provided."}), 400

    try:
        frame = decode_image(image_data)

        # Crear un directorio para la clase si no existe
        class_dir = os.path.join(DATA_DIR_PALABRAS, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Generar un nombre de archivo basado en la cantidad de imágenes en la carpeta de la clase
        image_count = len(os.listdir(class_dir))
        image_name = f"{image_count + 1}.jpg"
        image_path = os.path.join(class_dir, image_name)

        # Guardar la imagen en el directorio de la clase
        cv2.imwrite(image_path, frame)

        return jsonify({"message": f"Palabras img collected for class {class_name}", "image_path": image_path}), 200
            
    except Exception as e:
        return jsonify({"message": "Error al recolectar imágenes de palabras: " + str(e)}), 500

@app.route('/create-dataset/abecedario', methods=['POST'])
def create_abecedario_dataset():
    collector = DataCollector(DATA_DIR_ABECEDARIO)
    collector.collect_abecedario_data()
    return jsonify({"message": "Dataset del abecedario creado con éxito."}), 200

@app.route('/normalizar/palabras', methods=['POST'])
def create_palabras_dataset():
    normalize_class.process_all_words()
    return jsonify({"message": "Dataset de palabras creado con éxito."}), 200

# Nuevos endpoints para entrenar modelos
@app.route('/train-model/abecedario', methods=['POST'])
def train_abecedario_model():
    try:
        model_trainer.train_abecedario_model()  # Llama al método de entrenamiento
        return jsonify({"message": "Modelo del abecedario entrenado con éxito."}), 200
    except Exception as e:
        return jsonify({"message": "Error al entrenar el modelo del abecedario: " + str(e)}), 500

@app.route('/keypoints-palabras/palabras', methods=['POST'])
def train_palabras_model():
    try:
        keypoint_creator.create_keypoints() 
        return jsonify({"message": "Keypoints creados con éxito."}), 200
    except Exception as e:
        return jsonify({"message": "Error keypoints de palabras: " + str(e)}), 500

@app.route('/run-model/abecedario', methods=['POST'])
def run_abecedario_model():
    model_handler.run_abecedario_model()
    return jsonify({"message": "Modelo de abecedario iniciado."}), 200

@app.route('/training-model/palabras', methods=['POST'])
def trainning_palabras_model():
    training_model.training_model()
    return jsonify({"message": "Modelo de palabras entrenado."}), 200


if __name__ == '__main__':
    app.run(debug=True)
