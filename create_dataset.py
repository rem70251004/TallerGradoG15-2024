import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

class DataCollector:
    def __init__(self, data_dir):
        self.DATA_DIR = data_dir
        self.data = []
        self.labels = []
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
        self.model = None  # Inicializa tu modelo aquí

    def _process_images_abecedario(self):
        for dir_ in os.listdir(self.DATA_DIR):
            dir_path = os.path.join(self.DATA_DIR, dir_)
            if not os.path.isdir(dir_path):
                continue

            for img_path in os.listdir(dir_path):
                img_full_path = os.path.join(dir_path, img_path)
                
                img = cv2.imread(img_full_path)
                if img is None:
                    print(f"Error loading image {img_full_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)

                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:  # Solo una mano
                    hand_landmarks = results.multi_hand_landmarks[0]
                    data_aux = []
                    x_ = [landmark.x for landmark in hand_landmarks.landmark]
                    y_ = [landmark.y for landmark in hand_landmarks.landmark]

                    # Normaliza las coordenadas y agrega variaciones
                    for i in range(len(hand_landmarks.landmark)):
                        normalized_x = (x_[i] - min(x_)) + np.random.uniform(-0.05, 0.05)  # Agrega variación
                        normalized_y = (y_[i] - min(y_)) + np.random.uniform(-0.05, 0.05)  # Agrega variación
                        data_aux.append(normalized_x)
                        data_aux.append(normalized_y)

                    self.data.append(data_aux)
                    self.labels.append(dir_)

    def _process_images_palabras(self):
        for dir_ in os.listdir(self.DATA_DIR):
            dir_path = os.path.join(self.DATA_DIR, dir_)
            if not os.path.isdir(dir_path):
                continue

            for img_path in os.listdir(dir_path):
                img_full_path = os.path.join(dir_path, img_path)
                
                img = cv2.imread(img_full_path)
                if img is None:
                    print(f"Error loading image {img_full_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)

                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:  # Dos manos
                    accumulated_data = []

                    for hand_landmarks in results.multi_hand_landmarks:
                        # Obtener las coordenadas x e y de cada mano
                        x_ = [landmark.x for landmark in hand_landmarks.landmark]
                        y_ = [landmark.y for landmark in hand_landmarks.landmark]
  # Imprimir las coordenadas de cada mano
                        print(f"Hand Coordinates (x): {x_}")
                        print(f"Hand Coordinates (y): {y_}")
                        # Normalizar y agregar variaciones a las coordenadas
                        for i in range(len(hand_landmarks.landmark)):
                            normalized_x = (x_[i] - min(x_)) + np.random.uniform(-0.05, 0.05)  # Variación para x
                            normalized_y = (y_[i] - min(y_)) + np.random.uniform(-0.05, 0.05)  # Variación para y
                            accumulated_data.append(normalized_x)
                            accumulated_data.append(normalized_y)

                    # Verificar que se han almacenado las coordenadas de ambas manos (84 elementos)
                    if len(accumulated_data) == 84:  # 21 puntos de referencia * 2 manos * 2 coordenadas (x, y)
                        self.data.append(accumulated_data)
                        self.labels.append(dir_)

    def collect_abecedario_data(self):
        self.DATA_DIR = './data_abecedario'
        self._process_images_abecedario()
        self._save_data('data_abecedario.pickle')

    def collect_palabras_data(self):
        self.DATA_DIR = './data_palabras'
        self._process_images_palabras()
        self._save_data('data_palabras.pickle')

    def _save_data(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'data': self.data, 'labels': self.labels}, f)
