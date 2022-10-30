import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
# setup mp
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.8,
                            model_name='Shoe') as objectron:
    while cap.isOpened():
        start = time.time()

        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame.flags.writeable = False
        results = objectron.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(frame, detected_object.rotation, detected_object.translation)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime

        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        cv2.imshow('MediaPipe', frame)

        if cv2.waitKey(1) == ord('a'):
            break

    cap.release()
    cv2.destroyAllWindows()
