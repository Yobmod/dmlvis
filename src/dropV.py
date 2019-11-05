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

from typing import List, Tuple, cast, Union
from mytypes import videoType, imageType, floatArray  # , colorType, PILImage


def process_video(vid: videoType) -> np.ndarray:
    fps: int = vid.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    # frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # duration = frameCount / fps  # seconds
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
            angle, w, h = process_image(frame, "", save_images=False, verbosity=0)

            key: int = cv2.waitKey(10) & 0xFF
            start = False

            if key & 0xFF == ord("q"):  # quit camera if 'q' key is pressed
                break
            elif key & 0xFF == ord("s"):
                print('S key pressed - start processing')  # save the frame
                start = True
            elif key & 0xFF == ord("e"):
                print('E key pressed - end processing')  # save the frame
                start = False

            while start:
                CA_list.append(angle)
                W_list.append(w)
                H_list.append(h)
                frame_time = frame_no / fps
                frame_nos.append(frame_no)
                vid_times.append(frame_time)
                frame_no += 1

        else:
            break  # break if problem with a frame
    else:
        print("Error: Video not opened...")

    vid.release()
    masked_vid.release()
    cv2.destroyAllWindows()

    from scipy.signal import medfilt

    CA_filtered = medfilt(CA_list, kernel_size=7)  # kernal_size is length of sub-data to use as median. Bigger == more filtered
    CA_smoothed = savitzky_golay(CA_filtered, window_size=251, order=3)
    assert len(CA_smoothed) == len(CA_list)

    try:
        CA_array: np.ndarray = cast(np.ndarray, np.array((frame_nos, vid_times, W_list, H_list,
                                                          CA_list, CA_filtered, CA_smoothed), dtype=float).transpose())
    except ValueError:
        print("Not enough data for smoothing. Leave processing for long (e.g. 10+ seconds)")
        raise
    else:
        return CA_array


def print_video_data(data: np.ndarray,
                     out_path: Union[str, Path] = None,
                     out_filename: str = "out",
                     show_raw: bool = False,
                     show_smooth: bool = True
                     ) -> None:
    # set up save paths
    if out_path is None:
        out_folder_path = Path.cwd() / "out"
    elif isinstance(out_path, str):
        out_folder_path = Path(out_path)
    else:
        out_folder_path = out_path

    out_folder_path.mkdir(exist_ok=True)
    out_file = out_folder_path.joinpath(out_filename + ".csv")
    graph_file = out_folder_path.joinpath(out_filename + ".png")

    # sav data array
    headers = "frame, time, width, height, contact angles, CA_filtered, CA_smoothed"
    np.savetxt(out_file, data, delimiter=",", header=headers)  # np required by cv2, so may as well use it
    print(f"Results file saved to {out_file}")

    vid_time = data[:, 1]
    raw_data = data[:, 4]
    smoothed_data = data[:, 6]

    # plot and save graphs
    if show_smooth:
        plt.plot(vid_time, smoothed_data)
        plt.ylim(0, 90)
        plt.suptitle('Contact angle vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Contact angle (\u00B0)')
        plt.savefig(graph_file, format="png", bbox_inches="tight", tranparent=True)
        print(f"Graph saved to {graph_file}")

    if show_raw:
        plt.plot(vid_time, raw_data)
        plt.ylim(0, 90)
        plt.suptitle('Contact angle vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Contact angle (\u00B0)')
        plt.savefig(f"raw_{graph_file}", format="png", bbox_inches="tight", tranparent=True)
        print(f"Graph saved to {graph_file}")
        # TODO: plot graph, then optionally add raw data and smoothed


def print_video_data_from_csv(file_in: Union[str, Path],
                              out_path: Union[str, Path] = None,
                              out_filename: str = "out",
                              show_raw: bool = False,
                              show_smooth: bool = False,
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
    print_video_data(data_array, out_path, out_filename, show_raw, show_smooth)


def process_video_from_path(path: Union[str, Path]) -> np.ndarray:

    if isinstance(path, str):
        path = Path(path)

    video_path = path.resolve()

    if video_path.exists():
        print(f"Video loaded from {video_path}")
        vid: videoType = cv2.VideoCapture(str(video_path))
    else:
        sys.exit("Error: Video file not found...")

    data_array = process_video(vid)
    return data_array


if __name__ == "__main__":
    video_path = Path(R'..\data\thoria 300c dry 70.avi')
    video_stem = video_path.stem

    data = process_video_from_path(video_path)
    print_video_data(data, out_path=R'..\data\out', out_filename=video_stem)
