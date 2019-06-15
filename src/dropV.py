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

from dmlutils import get_outlined_image, crop_outlined_image, get_contour_lims, calc_contact_angle, get_image_skew, save_image_groups  # set_res

from typing import List, Tuple
from mytypes import videoType, imageType, floatArray  # , colorType, PILImage



video_title = R'.\tests\test.mp4'
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

raw_frames: List[imageType] = []

frame_no = 0

while vid.isOpened():  # and img.any():

    frame_out: Tuple[bool, imageType] = vid.read()
    # (ret, frame) = frame_out  ## loses typings :/
    ret = frame_out[0]
    frame = frame_out[1]

    if ret is True:

        cv2.imshow('image window', frame)
        raw_frames.append(frame)
        frame_no += 1
        # print(frame_no)
        if frame_no > 200:
            break

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

print(f"Number of frames = {len(raw_frames)}")


edged: List[imageType] = [get_outlined_image(frame) for frame in raw_frames]
SKEW_list: List[float] = [get_image_skew(edged_frame) for edged_frame in edged]


"""
fixed_frames = []
for num, frame in enumerate(edged):
    skew = SKEW_list[num]
    fixed_frame = imutils.rotate_bound(frame, -skew)
    fixed_frames.append(fixed_frame)
print(len(fixed_frames))
"""
fixed_frames = [imutils.rotate_bound(frame, -SKEW_list[num]) for num, frame in enumerate(edged)]


"""
CA_list: List[float] = []
for edged_frame in fixed_frames:
    (x, y, w, h) = get_contour_lims(edged_frame)
    ang = calc_contact_angle(w, h)
    CA_list.append(ang)
"""
contour_lims = [get_contour_lims(frame)[2:] for frame in fixed_frames]
CA_list = [calc_contact_angle(w, h) for (w, h) in contour_lims]
widths = [w for (w, h) in contour_lims]
heights = [h for (w, h) in contour_lims]

vid_times = [frame_no / fps for frame_no in range(len(raw_frames))]
masked_fixed_frames: List[imageType] = [crop_outlined_image(frame) for frame in fixed_frames]
# masked_vid.write(masked)

# print(CA_list)
assert len(CA_list) == frame_no
assert len(contour_lims) == frame_no
assert len(raw_frames) == frame_no

"""
CA_array: floatArray = np.zeros(len(CA_list) * 5)

for x in CA_list:
    CA_array[frame_no] = calc_contact_angle(w, h)
    # CA_array[frame_no, 0] = ang
    # CA_array[frame_no, 1] = w
    # CA_array[frame_no, 2] = h
"""


CA_array = np.array((vid_times, CA_list, widths, heights), dtype=float).transpose()
# CA_array.T
# print(CA_array)


# TODO: Save maked_frames, maked masked video?
# save raw, edged, masked
# save_image_groups(raw_frames)

out_folder_path = Path(video_folder / "out")
out_folder_path.mkdir(exist_ok=True)
out_file = out_folder_path.joinpath(video_stem + ".csv")
graph_file = out_folder_path.joinpath(video_stem + ".png")

headers = "time, contact angles, width, height"
np.savetxt(out_file, CA_array, delimiter=",", header=headers)  # np required by cv2, so may as well use it
print(f"Results file saved to {out_file}")

plt.plot(vid_times, CA_list)
plt.ylim(0, 90)
plt.suptitle('Contact angle vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Contact angle (\u00B0)')
plt.savefig(graph_file, format="png", bbox_inches="tight", tranparent=True)
print(f"Graph saved to {graph_file}")
# plt.show()
