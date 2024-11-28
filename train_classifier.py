from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from collections import Counter
from sklearn.svm import SVC

class ModelTrainer:
    def _train_model(self, data_path, model_path, num_features):
        try:
            # Cargar datos
            with open(data_path, 'rb') as file:
                data_dict = pickle.load(file)
                
            data = np.asarray(data_dict['data'])
            labels = np.asarray(data_dict['labels'])
            
            # Revisar el contenido
            print("Datos cargados, longitud:", len(data), "Longitud de labels:", len(labels))

            # Normalizar datos
            max_length = max(len(sample) for sample in data_dict['data'])
            normalized_data = [np.pad(sample, (0, max_length - len(sample)), 'constant') for sample in data_dict['data']]
            
            # Filtrar clases con al menos 2 muestras
            label_counts = Counter(labels)
            valid_labels = [label for label, count in label_counts.items() if count > 1]
            filtered_data = [normalized_data[i] for i in range(len(data)) if labels[i] in valid_labels]
            filtered_labels = [labels[i] for i in range(len(labels)) if labels[i] in valid_labels]
            
            # Convertir a numpy arrays
            filtered_data = np.asarray(filtered_data)
            filtered_labels = np.asarray(filtered_labels)

            # Verifica que los datos no están vacíos
            if len(filtered_data) == 0 or len(filtered_labels) == 0:
                print("Error: no hay datos suficientes para entrenar.")
                return

            # Dividir datos
            x_train, x_test, y_train, y_test = train_test_split(filtered_data, filtered_labels, test_size=0.2, stratify=filtered_labels)

            # Crear y entrenar el modelo
            # Cambia el modelo según tus necesidades
            if num_features == 42:  # Modelo para el abecedario (una mano)
                self.model = SVC(kernel='linear')  # Puedes cambiar el kernel según tus necesidades
            elif num_features == 84:  # Modelo para palabras (dos manos)
                self.model = RandomForestClassifier(n_estimators=100,random_state=42)  # Usa RandomForest para palabras

            self.model.fit(x_train, y_train)
            
            # Predecir y calcular precisión
            y_predict = self.model.predict(x_test)
            score = accuracy_score(y_test, y_predict)
            print(f"{score * 100}% de precisión en las muestras de prueba")

            # Guardar modelo
            with open(model_path, 'wb') as f:
                pickle.dump({'model': self.model}, f)

        except Exception as e:
            print(f"Error durante el entrenamiento: {str(e)}")

    def train_abecedario_model(self):
        # Entrena el modelo para el abecedario (42 características)
        self._train_model(data_path='./data_abecedario.pickle', model_path='model_abecedario.p', num_features=42)

    def train_palabras_model(self):
        # Entrena el modelo para palabras (84 características)
        self._train_model(data_path='./data_palabras.pickle', model_path='model_palabras.p', num_features=84)
