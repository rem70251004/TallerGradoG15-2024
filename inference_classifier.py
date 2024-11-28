import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe import solutions
from LSB.helpers import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from LSB.constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH, WORDS_JSON_PATH 
from datetime import datetime
from LSB.text_to_speech import text_to_speech
from LSB.evaluate_model import *
import json
class ModelHandler:
    def __init__(self):
        # Cargar modelos entrenados
        self.model_abecedario = None
        self.model_palabras = None

        # Inicializar Mediapipe para el reconocimiento de manos
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Obtener etiquetas de las carpetas
        self.labels_dict_abecedario = self.get_labels_from_folders('./data_abecedario')
        self.labels_dict_palabras = self.get_labels_from_folders('./data_palabras')

        # Configurar captura de video
        self.cap = cv2.VideoCapture(0)

    def get_labels_from_folders(self, data_dir):
        labels = {}
        for folder in os.listdir(data_dir):
            labels[folder] = folder  # Clave y valor son iguales
        return labels

    def generate_frames(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                print("No se pudo leer el cuadro de la cámara.")  # Mensaje de depuración
                break
            else:
                try:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    print(f"Error al codificar el frame: {str(e)}")

    def capture_samples(self,word_name, margin_frame=1, min_cant_frames=5, delay_frames=3):
        path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
        '''
        ### CAPTURA DE MUESTRAS PARA UNA PALABRA
        Recibe como parámetro la ubicación de guardado y guarda los frames
        
        `path` ruta de la carpeta de la palabra \n
        `margin_frame` cantidad de frames que se ignoran al comienzo y al final \n
        `min_cant_frames` cantidad de frames minimos para cada muestra \n
        `delay_frames` cantidad de frames que espera antes de detener la captura después de no detectar manos
        '''
      
        create_folder(path)
        if os.path.exists(WORDS_JSON_PATH ):
            with open(WORDS_JSON_PATH , 'r') as json_file:
               data = json.load(json_file)
        else:
            data = {"word_ids": []}  # Estructura JSON inicial si el archivo no existe

        # Agregar el nuevo `word_name` a la lista `word_ids` si aún no está en ella
        if word_name not in data["word_ids"]:
            data["word_ids"].append(word_name)

        # Guardar los cambios en el archivo JSON
        with open(WORDS_JSON_PATH , 'w') as json_file:
            json.dump(data, json_file, indent=4)
    
        count_frame = 0
        frames = []
        fix_frames = 0
        recording = False
    
        with solutions.holistic.Holistic() as holistic_model:
   
        
         while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            image = frame.copy()
            results = mediapipe_detection(frame, holistic_model)
            
            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    cv2.putText(image, 'Capturando...', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                    frames.append(np.asarray(frame))
            else:
                if len(frames) >= min_cant_frames + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    frames = frames[: - (margin_frame + delay_frames)]
                    today = datetime.now().strftime('%y%m%d%H%M%S%f')
                    output_folder = os.path.join(path, f"sample_{today}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)
                
                recording, fix_frames = False, 0
                frames, count_frame = [], 0
                cv2.putText(image, 'Listo para capturar...', FONT_POS, FONT, FONT_SIZE, (0,220, 100))
            
            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
            
            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

       

    
    def run_model_abecedario(self):
        hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
        
        while True:
            data_aux = []
            x_hand = []
            y_hand = []

            # Procesar el frame
            ret, frame = self.cap.read()
            if not ret:
                continue  # Asegúrate de que el frame se ha leído correctamente

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            
            # Verifica si solo se detecta una mano
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                # Procesar las coordenadas de la mano
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_hand.append(x)
                    y_hand.append(y)

                # Normalizar las coordenadas de la mano
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(x_hand[i] - min(x_hand))
                    data_aux.append(y_hand[i] - min(y_hand))

                # Verifica que data_aux tenga las características correctas
                if len(data_aux) == 42:  # 21 puntos de referencia * 2 (x, y)
                    prediction = self.model_abecedario.predict([np.asarray(data_aux)])
                    predicted_character = self.labels_dict_abecedario[prediction[0]]

                    # Dibuja el rectángulo y el texto
                    x1 = int(min(x_hand) * W) - 10
                    y1 = int(min(y_hand) * H) - 10
                    x2 = int(max(x_hand) * W) - 10
                    y2 = int(max(y_hand) * H) - 10

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            # Codificar el marco como imagen JPEG y generarlo como parte del stream
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def run_model_palabras(self,src=None, threshold=0.4, margin_frame=1, delay_frames=3):
        kp_seq, sentence = [], []
        word_ids = get_word_ids(WORDS_JSON_PATH)
        model = models.load_model(MODEL_PATH)
        count_frame = 0
        fix_frames = 0
        recording = False
        with solutions.holistic.Holistic() as holistic_model:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break

                results = mediapipe_detection(frame, holistic_model)
                
                # TODO: colocar un máximo de frames para cada seña,
                # es decir, que traduzca incluso cuando hay mano si se llega a ese máximo.
                if there_hand(results) or recording:
                    recording = False
                    count_frame += 1
                    if count_frame > margin_frame:
                        kp_frame = extract_keypoints(results)
                        kp_seq.append(kp_frame)
                
                else:
                    if count_frame >= MIN_LENGTH_FRAMES + margin_frame:
                        fix_frames += 1
                        if fix_frames < delay_frames:
                            recording = True
                            continue
                        kp_seq = kp_seq[: - (margin_frame + delay_frames)]
                        kp_normalized = normalize_keypoints(kp_seq, int(MODEL_FRAMES))
                        res = model.predict(np.expand_dims(kp_normalized, axis=0))[0]
                        
                        print(np.argmax(res), f"({res[np.argmax(res)] * 100:.2f}%)")
                        if res[np.argmax(res)] > threshold:
                            word_id = word_ids[np.argmax(res)].split('-')[0]
                            
                            sent = words_text.get(word_id)
                            sentence.insert(0, sent)
                            text_to_speech(sent) # ONLY LOCAL (NO SERVER)
                    
                    recording = False
                    fix_frames = 0
                    count_frame = 0
                    kp_seq = []
                
                if not src:
                    cv2.rectangle(frame, (0, 0), (640, 35), (245, 117, 16), -1)
                    cv2.putText(frame, ' | '.join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))
                    
                draw_keypoints(frame, results)
                cv2.imshow('Traductor LSP', frame)

            # Codificar el marco como imagen JPEG y generarlo como parte del stream
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def run_abecedario_model(self):
        self.model_abecedario = pickle.load(open('./model_abecedario.p', 'rb'))['model']
        return self.run_model_abecedario()

    def run_palabras_model(self):
        return self.run_model_palabras()
