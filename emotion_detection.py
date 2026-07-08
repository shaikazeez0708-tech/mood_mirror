from deepface import DeepFace
import cv2

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    result = DeepFace.analyze(
        frame,
        actions=['emotion'],
        enforce_detection=False
    )

    emotion = result[0]['dominant_emotion']

    print("Detected Emotion:", emotion)

    cv2.imshow("MoodMirror AI", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
