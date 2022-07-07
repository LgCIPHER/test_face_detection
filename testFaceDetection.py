import cv2 as cv
from matplotlib import image
import mediapipe as mp

import os.path as path

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def webcam():
    cap = cv.VideoCapture(0)

    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)

            # Flip the image horizontally for a selfie-view display.
            cv.imshow('MediaPipe Face Detection', cv.flip(image, 1))

            # Press "Esc" to end the process
            if cv.waitKey(5) & 0xFF == 27:
                break
    cap.release()


def image():
    # Path to images folder
    dir_path = path.dirname(path.realpath(__file__))
    images_folder_path = path.join(dir_path, "images/")

    # Open up the image selected
    image_path = path.join(images_folder_path, "lul.png")

    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:

        image = cv.imread(image_path)

        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        # Draw face detections of each face.
        if not results.detections:
            print("Something is wrong...")

        annotated_image = image.copy()

        for detection in results.detections:
            print('Nose tip:')
            print(mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            mp_drawing.draw_detection(annotated_image, detection)

        cv.imshow("Result", annotated_image)

        cv.waitKey(0)


if __name__ == "__main__":
    # webcam()
    image()
