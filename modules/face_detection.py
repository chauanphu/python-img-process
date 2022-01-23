import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face = mp.solutions.face_detection

cap = cv.VideoCapture(0)

if __name__ == '__main__':
    with mp_face.FaceDetection(
        model_selection = 0,
        min_detection_confidence = 0.5,
    ) as face:
        while cap.isOpened():
            success, img = cap.read()

            if not success:
                print("Error ! Can't open camera")
                continue
            
            # Flip image
            img = cv.flip(img, 1)

            #Detect pose
            img.flags.writeable = False
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = face.process(img)

            # Draw rectangles
            img.flags.writeable = True
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(
                        img, detection)

            # Show stream video
            cv.imshow("Webcam", img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break   
        cap.release()
        cv.destroyAllWindows()