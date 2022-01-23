import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv.VideoCapture(0)

if __name__ == '__main__':
    with mp_pose.Pose(
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    ) as pose:
        while cap.isOpened():
            success, img = cap.read()

            if not success:
                print("Error ! Can't open camera")
                continue
            
            #Detect pose
            img.flags.writeable = False
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = pose.process(img)

            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS)
                
            # Flip image
            img = cv.flip(img, 1)

            # Show stream video
            cv.imshow("Webcam", img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break   
        cap.release()
        cv.destroyAllWindows()