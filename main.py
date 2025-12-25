import cv2
import mediapipe as mp
import time

print("Starting MediaPipe and Emotion Recognition...")

# MediaPipe yüz landmark'ı için ayarlar
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,  # Tek görüntü modu
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)

# Duygu tanıma fonksiyonu
def detect_emotion(blendshapes):
    scores = {bs.category_name: bs.score for bs in blendshapes}
    
    # Duygu skorları
    happy = max(scores.get('mouthSmileLeft', 0), scores.get('mouthSmileRight', 0))
    sad = max(scores.get('mouthFrownLeft', 0), scores.get('mouthFrownRight', 0))
    angry = max(scores.get('browDownLeft', 0), scores.get('browDownRight', 0), scores.get('noseSneerLeft', 0), scores.get('noseSneerRight', 0))
    surprised = max(scores.get('eyeWideLeft', 0), scores.get('eyeWideRight', 0), scores.get('jawOpen', 0))
    fear = max(scores.get('browInnerUp', 0), scores.get('eyeWideLeft', 0), scores.get('eyeWideRight', 0))
    disgust = max(scores.get('noseSneerLeft', 0), scores.get('noseSneerRight', 0), scores.get('mouthUpperUpLeft', 0), scores.get('mouthUpperUpRight', 0))
    
    emotions = {
        'Happy': happy,
        'Sad': sad,
        'Angry': angry,
        'Surprised': surprised,
        'Fear': fear,
        'Disgust': disgust
    }
    
    max_emotion = max(emotions, key=emotions.get)
    confidence = emotions[max_emotion]
    
    if confidence > 0.2:
        return max_emotion, confidence
    else:
        return 'Neutral', 0.0

# Yüz landmark'ını başlat
face_landmarker = FaceLandmarker.create_from_options(options)

# Kamera aç
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened! Please check permissions.")
    exit()

print("Camera opened. Press ESC to exit.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read camera frame.")
        break

    # Görüntüyü RGB'ye çevir
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Yüzü işle
    results = face_landmarker.detect(mp_image)

    # Yüz bulunduysa çiz ve duygu tanı
    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            # Bounding box hesapla
            x_coords = [int(lm.x * frame.shape[1]) for lm in face_landmarks]
            y_coords = [int(lm.y * frame.shape[0]) for lm in face_landmarks]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            
            # Landmark'ları çiz (noktalar)
            for lm in face_landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Duygu tanı
        if results.face_blendshapes:
            emotion, confidence = detect_emotion(results.face_blendshapes[0])
            cv2.putText(frame, f'Emotion: {emotion} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # FPS hesapla
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Görüntüyü göster
    cv2.imshow("Face Mesh and Emotion Recognition", frame)

    # ESC ile çık
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Temizlik
face_landmarker.close()
cap.release()
cv2.destroyAllWindows()

print("Program ended.")
