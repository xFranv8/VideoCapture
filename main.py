import cv2


def main():
    capture = cv2.VideoCapture(4)

    if not capture.isOpened():
        print("Error opening video stream or file")
        exit(1)

    while True:
        ret, frame = capture.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            exit(1)

        cv2.imshow("Cam", frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
