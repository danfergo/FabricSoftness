import cv2
import pathlib
import numpy as np

def captureCategoryImages(n_elems, category, to_folder=False, i=0):

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('2018-11-14-141611.webm')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while (i < n_elems):

        ret, frame = cap.read()
        key = cv2.waitKey(1000) & 0xFF
        gray = frame  # cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if key == ord('q'):
            break
        elif key == ord('c') and to_folder:
            status = cv2.imwrite(str(to_folder) + str(i) + '.png', gray)
            if status:
                print("Image written to file-system : ", str(i))
                i += 1

        # Display the resulting frame
        cv2.imshow('frame', gray)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def waitNext():
    print('Next (press n) ...')
    while True:
        m = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imshow('frame', m)
        if (cv2.waitKey(1000) & 0xFF) != ord('n'):
            cv2.destroyAllWindows()
            break

def captureImages(n_elems, categories, path):
    pathlib.Path(path).mkdir(parents=True)

    print('Capturing average (press c to capture) ...')
    captureCategoryImages(3, 'background', path + '/' + 'background')
    waitNext()


    for category in categories:
        print('Capturing ' + category + '(press c to capture) ...')
        captureCategoryImages(n_elems, category, path + '/' + category)
        waitNext()

    print('Capturing average...')
    captureCategoryImages(3, 'background', path + '/' + 'background', 3)


if __name__ == '__main__':
    captureImages(1500, ['stiff', 'soft'], 'images/test/')
