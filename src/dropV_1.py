"""scripts to preocess droplet video file, saving video output and csv of relavent data (frame_no, time, angle, height, width)
Required modules: dmlutils, mytypes
Dependencies: CV2, numpy, Pillow"""
# -*- coding: utf-8 -*-
import cv2
# from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
import numpy as np
import sys
import imutils
import matplotlib.pyplot as plt

from dmlutils import (get_outlined_image, crop_outlined_image, get_contour_lims, 
                      calc_contact_angle, get_image_skew, save_image_groups, savitzky_golay)  # set_res
from dropI import process_image

from typing import List, Tuple
from mytypes import videoType, imageType, floatArray  # , colorType, PILImage



video_title = R'..\data\thoria 300c dry 70.avi'

video_path = Path(video_title).resolve()
video_folder = video_path.parent  # .resolve()  # .resolve()  ?
video_stem = video_path.stem  # .anchor (.drive .root) .parent .name (.stem .suffix)

if video_path.exists():
    print(f"Video loaded from {video_path}")
    vid: videoType = cv2.VideoCapture(video_title)
else:
    sys.exit("Error: Video file not found...")

fps: int = vid.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frameCount / fps  # seconds
# print(duration)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
masked_vid: videoType = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

CA_list: List[float] = []
W_list: List[float] = []
H_list: List[float] = []
vid_times: List[float] = []
frame_nos: List[int] = []
frame_no = 0

while vid.isOpened():  # and img.any():

    frame_out: Tuple[bool, imageType] = vid.read()
    # (ret, frame) = frame_out  ## loses typings :/
    ret = frame_out[0]
    frame = frame_out[1]

    if ret is True:

        cv2.imshow('image window', frame)
        angle, w, h = process_image(frame, video_stem, save_images=False, verbosity=0)
        CA_list.append(angle)
        W_list.append(w)
        H_list.append(h)
        frame_time = frame_no / fps
        frame_nos.append(frame_no)
        vid_times.append(frame_time)
        frame_no += 1


        key: int = cv2.waitKey(10) & 0xFF

        if key & 0xFF == ord("q"):  # quit camera if 'q' key is pressed
            break
        elif key & 0xFF == ord("s"):
            print('S key pressed - saved frame')  # save the frame
    else:
        break  # break if problem with a frame

else:
    print("Error: Video not opened...")

vid.release()
masked_vid.release()
cv2.destroyAllWindows()


# make array of the 5 lists
CA_smoothed = savitzky_golay(CA_list, window_size=251, order=3)
CA_smoothed = [x for x in CA_smoothed]
CA_array = np.array((frame_nos, vid_times, W_list, H_list, CA_list, CA_smoothed), dtype=float).transpose()

# set up save paths
out_folder_path = Path(video_folder / "out")
out_folder_path.mkdir(exist_ok=True)
out_file = out_folder_path.joinpath(video_stem + ".csv")
graph_file = out_folder_path.joinpath(video_stem + ".png")

# sav data array
headers = "frame, time, width, height, contact angles,"
np.savetxt(out_file, CA_array, delimiter=",", header=headers)  # np required by cv2, so may as well use it
print(f"Results file saved to {out_file}")

# plot and save graphs
plt.plot(vid_times, CA_list_smoothed)
plt.ylim(0, 90)
plt.suptitle('Contact angle vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Contact angle (\u00B0)')
plt.savefig(graph_file, format="png", bbox_inches="tight", tranparent=True)
print(f"Graph saved to {graph_file}")
# plt.show()
