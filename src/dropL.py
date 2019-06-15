"""."""
import cv2

from dmlutils import set_res, save_image_groups

import typing as t
from mytypes import imageType


cap = cv2.VideoCapture(0)

if cap.isOpened() is True:
    print("initialising camera")
else:
    print("no camera found")
    cap = cv2.VideoCapture(1)

set_res(cap, 480)
raw_frames = []

while(cap.isOpened()):
    frame_out: t.Tuple[bool, imageType] = cap.read()
    # (ret, frame) = frame_out  ## loses typings :/
    ret = frame_out[0]
    frame = frame_out[1]

    if ret is True:
        # Our operations on the frame come here
        # gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 0=vert, 1=horiz, -1=both       # Display the resulting frame
        # vflipped: imageType = cv2.flip(frame, 0)
        # 0=vert, 1=horiz, -1=both       # Display the resulting frame
        # hflipped: imageType = cv2.flip(frame, 1)

        # blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        # # edged = cv2.Canny(gray, 20, 100)

        cv2.imshow('image window', frame)
        # cv2.imshow('frame1', gray)
        # cv2.imshow('frame2', vflipped)
        # cv2.imshow('frame2', hflipped)

        key = cv2.waitKey(1)  # wait 1 ms after each frame is shown to capture any input

    # quit camera if 'q' key is pressed
        if key & 0xFF == ord("q"):
            break
        elif key & 0xFF == ord("s"):
            # save the frame
            raw_frames.append(frame)
            print('S key pressed - saved frame')
    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

save_image_groups(raw_frames)
