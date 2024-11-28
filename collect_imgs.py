import os
import base64
import numpy as np
import cv2

class ImageCollector:
    def __init__(self, data_dir, dataset_size=100):
        self.DATA_DIR = data_dir  # Este es el directorio donde se guardarán las clases
        self.dataset_size = dataset_size  # Siempre será 100

        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)

    def _collect_images(self, class_name, image_data):
        # Crear la carpeta dentro del directorio especificado
        class_dir = os.path.join(self.DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print(f'Collecting data for class {class_name} in directory: {class_dir}')

        # Decodificar la imagen de base64 a una imagen OpenCV
        image_bytes = base64.b64decode(image_data)
        image_np = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Verificar el número de imágenes existentes
        existing_images = len(os.listdir(class_dir))
        if existing_images < self.dataset_size:
            image_path = os.path.join(class_dir, f'{existing_images}.jpg')  # Nombres desde 0 hasta 99
            cv2.imwrite(image_path, frame)
            print(f'Image saved: {image_path}')
            return image_path
        else:
            print(f'Maximum number of images collected for class {class_name}.')
            return None

    def collect_imgs_abecedario(self, class_name, image_data):
        return self._collect_images(class_name, image_data)  # Recibe el nombre de la clase y los datos de la imagen

    def collect_imgs_palabras(self, class_name, image_data):
        return self._collect_images(class_name, image_data)  # Recibe el nombre de la clase y los datos de la imagen

    def release(self):
        cv2.destroyAllWindows()
