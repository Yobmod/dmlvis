"""scripts to process droplet video file, saving video output and csv of relavent data (frame_no, time, angle, height, width)
Required modules: dmlutils, mytypes
Dependencies: CV2, numpy, Pillow"""
# -*- coding: utf-8 -*-

from __future__ import annotations
import cv2
# from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
import numpy as np
import sys
import imutils
import matplotlib.pyplot as plt
from typing import Sequence, Collection
from typing_extensions import Literal

from dmlutils import (get_outlined_image, crop_outlined_image, get_contour_lims,
                      calc_contact_angle, get_image_skew, save_image_groups, savitzky_golay)  # set_res
from dropI import process_image

<<<<<<< HEAD
from typing import List, Tuple, Optional as Opt
from typing_extensions import Final as Fin
=======
from typing import List, Tuple, cast, Union
>>>>>>> 5f01ecdf9b78becb7faee78bf9680be3cecbb33e
from mytypes import videoType, imageType, floatArray  # , colorType, PILImage

from scipy.signal import medfilt, filtfilt, butter, ellip
from typing import NamedTuple


<<<<<<< HEAD
video_title = R'.\tests\test.mp4'
video_path = Path(video_title).resolve()
video_folder = video_path.parent  # .resolve()  # .resolve()  ?
video_stem = video_path.stem  # .anchor (.drive .root) .parent .name (.stem .suffix)

if video_path.exists():
    print(f"Video loaded from {video_path}")
    vid: videoType = cv2.VideoCapture(video_title)
else:
    sys.exit("Error: Video file not found...")

fps: Fin[int] = vid.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frameCount / fps  # seconds
# print(duration)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
masked_vid: videoType = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
=======
def process_video(vid: videoType, show: bool = True, manual: bool = False) -> np.ndarray:
    fps: int = vid.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    # frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # duration = frameCount / fps  # seconds
    # print(duration)
>>>>>>> 5f01ecdf9b78becb7faee78bf9680be3cecbb33e

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # masked_vid: videoType = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    CA_list: List[float] = []
    W_list: List[float] = []
    H_list: List[float] = []
    vid_times: List[float] = []
    frame_nos: List[int] = []
    frame_no = 0
    if manual:
        start = False
    else:
        start = True

    while vid.isOpened():  # and img.any():

        frame_out: Tuple[bool, imageType] = vid.read()
        # (ret, frame) = frame_out  ## loses typings :/
        ret = frame_out[0]
        frame = frame_out[1]

        if start:
            print(f'Frame processed: {frame_no}\r')

        if ret is True:

            if show:
                cv2.imshow('image window', frame)

            key: int = cv2.waitKey(10) & 0xFF

            if key & 0xFF == ord("q"):  # quit camera if 'q' key is pressed
                break
            elif key & 0xFF == ord("s"):
                print('S key pressed - start processing')
                start = True
            elif key & 0xFF == ord("e"):
                print('E key pressed - end processing')
                start = False

            if start:
                w, h, angle = process_image(frame, "", save_images=False, verbosity=0)
                CA_list.append(angle)
                W_list.append(w)
                H_list.append(h)
                frame_time = round(frame_no / fps, 2)
                frame_nos.append(frame_no)
                vid_times.append(frame_time)
                frame_no += 1

        else:
            break  # break if problem with a frame
    else:
        print("Error: Video not opened...")

    vid.release()
    # masked_vid.release()
    cv2.destroyAllWindows()

    # print(CA_list)
    if len(CA_list) <= 0:
        raise ValueError("No data produced...")

    try:
        CA_raw_array = cast('np.ndarray[float]', np.array((frame_nos,
                                                           vid_times,
                                                           W_list,
                                                           H_list,
                                                           CA_list
                                                           ), dtype=float))

    except ValueError:
        print("Not enough data for smoothing. Leave processing for long (e.g. 10+ seconds)")
        raise
    else:
        return CA_raw_array


def process_videodata(raw_data: 'np.ndarray[float]', flatten: Literal['high', 'med', 'low'] = 'med') -> 'np.ndarray[float]':
    """process raw_data from video (as a 5 row ndarray) and return a 9 column array"""

    # check if width (raw[2]) makes sense, else remove all data for that frame
    for i, num in enumerate(raw_data[2]):
        if num > 130:
            raw_data[2:7, i] = np.NaN

    # if CA greater than 80o remove
    for i, num in enumerate(raw_data[4]):
        if num > 80:
            raw_data[4, i] = np.NaN

    # drop the first minute
    for i, num in enumerate(raw_data[1]):
        if num <= 60:
            raw_data[1, i] = np.NaN

    # drop any row contain NaN width or CA. Shape is (5, xxx)
    raw_data = raw_data[:, ~np.any(np.isnan(raw_data), axis=0)]

    # check any data processed. Note CA_list has NaN replacements mutated above.
    if len(raw_data) < 5:
        raise ValueError("Not enough data ro process ...")

    CA_list = raw_data[4]
    if len(CA_list) <= 0:
        raise ValueError("No data produced...")

    if flatten == 'high':
        (b, a) = butter(1, 0.01)
    elif flatten == 'med':
        (b, a) = butter(1, 0.02)
    elif flatten == 'low':
        (b, a) = butter(1, 0.05)

    try:
        # median window filtered
        # kernal_size is length of sub-data used as median. Bigger == more filtered
        CA_medfilt: 'np.ndarray[float]' = medfilt(CA_list, kernel_size=11)
        assert len(CA_medfilt) == len(CA_list)

        # butterworth filter  # N 1-10 (higer = more), Wn 0.01-0.1 (smaller = more filter)
        CA_butterfilt: 'np.ndarray[float]' = filtfilt(b, a, CA_medfilt)  # padtype=xxx, padlen=None)
        assert len(CA_butterfilt) == len(CA_list)

        # savitzky_golay filter
        CA_smoothed: 'np.ndarray[float]' = savitzky_golay(CA_medfilt, window_size=101, order=3)
        assert len(CA_smoothed) == len(CA_list)

    except ValueError:
        print("Not enough data for smoothing. Leave processing for long (e.g. 10+ seconds)")
        raise

    data: 'np.ndarray[float]' = np.vstack((raw_data, CA_medfilt, CA_smoothed, CA_butterfilt)).transpose()
    return data


def print_video_data(data: 'np.ndarray[float]',
                     out_path: Union[str, Path] = None,
                     out_filename: str = "out",
                     combine: bool = True,
                     dots: Collection = "raw, medfilt",
                     lines: Collection = "smooth, butter",
                     sparse: Tuple[str, int] = ("raw, medfilt", 10)
                     ) -> None:
    """sparse only plots the xth value for given arrays"""

    # set up save paths
    if out_path is None:
        out_folder_path = Path.cwd() / "out"
    elif isinstance(out_path, str):
        out_folder_path = Path(out_path)
    else:
        out_folder_path = out_path

    out_folder_path.mkdir(exist_ok=True)
    out_file = out_folder_path.joinpath(out_filename + ".csv")

    # sav data array
    headers = "frame, time, width, height, contact angles, CA_medfiltered, CA_smoothed, CA_butterfilt"
    print(data.shape)
    assert len(headers.split(",")) == data.shape[1]  # horizontal
    np.savetxt(out_file, data, delimiter=",", header=headers)  # np required by cv2, so may as well use it
    print(f"Results file saved to {out_file}")

    vid_time: "np.ndarray[float]" = data[:, 1]
    raw_data: "np.ndarray[float]" = data[:, 4]
    medfilt_data: "np.ndarray[float]" = data[:, 5]
    smoothed_data: "np.ndarray[float]" = data[:, 6]
    butterfiltered: "np.ndarray[float]" = data[:, 7]

    # TODO: plot graph based on raw,
    # show raw as lines or dots or not
    # optional add filtered lines

    def plot_graph(data: "np.ndarray[float]", file_suffix: str, color: str = 'b', combine: bool = combine) -> None:
        from numpy.polynomial import polynomial as P
        lsf_coeffs = P.polyfit(vid_time, data, deg=5)
        lsf_data = P.polyval(vid_time, lsf_coeffs)
        assert len(lsf_data) == len(data)

        file_name = out_folder_path.joinpath(out_filename + file_suffix)
        plt.plot(vid_time, data, linestyle='None', color=color, marker="o", markersize=2.0)
        plt.plot(vid_time, lsf_data, linestyle='solid', linewidth=1, color=color, marker='')

        plt.ylim(0, 90)
        plt.suptitle('Contact angle vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Contact angle (\u00B0)')
        plt.savefig(file_name, format="png", bbox_inches="tight", tranparent=True)
        plt.close()
        print(f"Graph saved to {file_name}")

    # combined_graph = out_folder_path.joinpath(out_filename + "_combined.png")
    # plot and save graphs

    if "raw" in lines:
        plot_graph(data=raw_data, file_suffix="_raw.png")

    if "medfilt" in lines or "medfilt" in dots:
        plot_graph(data=medfilt_data, file_suffix="_medianfiltered.png", color='g')

    if "smooth" in lines:
        plot_graph(data=smoothed_data, file_suffix="_smoothed.png", color='m')

    if "butter" in lines:
        plot_graph(data=butterfiltered, file_suffix="_buttered.png", color="r")


def print_video_data_from_csv(file_in: Union[str, Path],
                              out_path: Union[str, Path] = None,
                              out_filename: str = "out",
                              *,
                              combine: bool = False,
                              lines: Collection[str] = "smooth, buttered",
                              dots: Collection[str] = "raw, medfilt",
                              delimiter: str = ','
                              ) -> None:
    """Given a .csv, .txt, .tsv, or .xlsx loads it as np.array and processes the data (re-saves as .csv and graph as .png)"""
    # numpy.loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None,
    # unpack=False, ndmin=0, encoding='bytes', max_rows=None)[source]
    if '.csv' in str(file_in):
        delim = ','
    elif '.tsv' in str(file_in):
        delim = '\t'
    else:
        delim = delimiter

    data_array = np.loadtxt(file_in, delimiter=delim)
    print_video_data(data_array, out_path, out_filename, combine=combine, lines=lines, dots=dots)


def process_video_from_path(path: Union[str, Path],
                            show: bool = True,
                            manual: bool = False
                            ) -> np.ndarray:

    if isinstance(path, str):
        path = Path(path)

    video_path = path.resolve()

    if video_path.exists():
        print(f"Video loaded from {video_path}")
        vid: videoType = cv2.VideoCapture(str(video_path))
    else:
<<<<<<< HEAD
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
masked_fixed_frames: List[Opt[imageType]] = [crop_outlined_image(frame) for frame in fixed_frames]
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
=======
        sys.exit("Error: Video file not found...")

    data_array = process_video(vid, show=show, manual=manual)
    return data_array


def process_videos_from_dirs(in_path: Union[str, Path],
                             out_path: Union[str, Path] = "",
                             *,
                             lines: Collection = "smooth, butter",
                             dots: Collection = "raw, medfilt",
                             flatten: Literal["low", "med", "high"] = "med",
                             show: bool = True,
                             manual: bool = False,
                             combine: bool = False,
                             ) -> None:
    """get all .avi files from folder and sub-folders as Paths and process each to out_path"""

    if isinstance(in_path, str):
        in_path = Path(in_path)

    if not out_path:
        out_path = in_path / "out"
    elif isinstance(out_path, str):
        out_path = Path(out_path)

    # get all .avi files from folder and sub-folders as Paths
    for video_path in in_path.glob('**/*.mp4'):
        video_stem = video_path.stem
        raw_data = process_video_from_path(video_path, show=show, manual=manual)
        processed_data = process_videodata(raw_data, flatten=flatten)
        print_video_data(processed_data, out_path=data_path / 'out', out_filename=video_stem, combine=combine, lines=lines, dots=dots)


if __name__ == "__main__":
    data_path = Path(R'..\data')
    process_videos_from_dirs(data_path, data_path / "out",
                             lines="raw, medfilt, butter, smooth",
                             dots="",
                             flatten='high',
                             show=True,
                             combine=False)

    # video_path = Path(R'..\data\thoria 300c dry 70.avi')
    # video_stem = video_path.stem
    # raw_data = process_video_from_path(video_path)
    # processed_data = process_videodata(raw_data, flatten='low')
    # print_video_data(processed_data, out_path=R'..\data\out', out_filename=video_stem, combine=True)
>>>>>>> 5f01ecdf9b78becb7faee78bf9680be3cecbb33e
