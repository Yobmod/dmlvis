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
from typing import Sequence, Collection
from typing_extensions import Literal

from dmlutils import (get_outlined_image, crop_outlined_image, get_contour_lims,
                      calc_contact_angle, get_image_skew, save_image_groups, savitzky_golay)  # set_res
from dropI import process_image

from typing import List, Tuple, cast, Union
from mytypes import videoType, imageType, floatArray  # , colorType, PILImage

from scipy.signal import medfilt, filtfilt, butter, ellip
from typing import NamedTuple


def process_video(vid: videoType, manual: bool = False) -> np.ndarray:
    fps: int = vid.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    # frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # duration = frameCount / fps  # seconds
    # print(duration)

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
                angle, w, h = process_image(frame, "", save_images=False, verbosity=0)
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
        # butterworth filter  # N 1-10 (higer = more), Wn 0.01-0.1 (smaller = more filter)
        CA_butterfilt: 'np.ndarray[float]' = filtfilt(b, a, CA_list, padlen=None)
        assert len(CA_butterfilt) == len(CA_list)

        CA_gustfilt: 'np.ndarray[float]' = filtfilt(b, a, CA_list, method="gust")
        assert len(CA_gustfilt) == len(CA_list)

        # median window filtered
        CA_medfilt: 'np.ndarray[float]' = medfilt(CA_list, kernel_size=7)  # kernal_size is length of sub-data used as median. Bigger == more filtered
        assert len(CA_medfilt) == len(CA_list)

        # savitzky_golay filter
        CA_smoothed: 'np.ndarray[float]' = savitzky_golay(CA_list, window_size=151, order=3)
        assert len(CA_smoothed) == len(CA_list)

    except ValueError:
        print("Not enough data for smoothing. Leave processing for long (e.g. 10+ seconds)")
        raise

    data: 'np.ndarray[float]' = np.vstack((raw_data, CA_medfilt, CA_smoothed, CA_butterfilt, CA_gustfilt)).transpose()
    return data


def print_video_data(data: 'np.ndarray[float]',
                     out_path: Union[str, Path] = None,
                     out_filename: str = "out",
                     combine: bool = True,
                     raw_dots: bool = False,
                     lines: Collection = "raw, medfilt, smooth, butter, gust",
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

    # sav data array
    headers = "frame, time, width, height, contact angles, CA_medfiltered, CA_smoothed, CA_butterfilt, CA_gustfilt"
    print(data.shape)
    assert len(headers.split(",")) == data.shape[1]  # horizontal
    np.savetxt(out_file, data, delimiter=",", header=headers)  # np required by cv2, so may as well use it
    print(f"Results file saved to {out_file}")

    vid_time = data[:, 1]
    raw_data = data[:, 4]
    medfilt_data = data[:, 5]
    smoothed_data = data[:, 6]
    butterfiltered = data[:, 7]
    gustfiltered = data[:, 8]

    # TODO: plot graph based on raw,
    # show raw as lines or dots or not
    # optional add filtered lines
    combined_graph = out_folder_path.joinpath(out_filename + "_combined.png")

    # plot and save graphs
    if "raw" in lines:
        if combine:
            plt.figure(0)
            file_name = combined_graph
        else:
            plt.figure(1)
            file_name = out_folder_path.joinpath(out_filename + "_raw.png")
        plt.plot(vid_time, raw_data)
        plt.ylim(0, 90)
        plt.suptitle('Contact angle vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Contact angle (\u00B0)')
        plt.savefig(file_name, format="png", bbox_inches="tight", tranparent=True)
        print(f"Graph saved to {file_name}")

    if "medfilt" in lines:

        if combine:
            file_name = combined_graph
            plt.figure(0)
        else:
            plt.figure(2)
        file_name = out_folder_path.joinpath(out_filename + "_medianfiltered.png")
        plt.plot(vid_time, medfilt_data)
        plt.ylim(0, 90)
        plt.suptitle('Contact angle vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Contact angle (\u00B0)')
        plt.savefig(file_name, format="png", bbox_inches="tight", tranparent=True)
        print(f"Graph saved to {file_name}")

    if "smooth" in lines:
        if combine:
            plt.figure(0)
            file_name = combined_graph
        else:
            plt.figure(3)
            file_name = out_folder_path.joinpath(out_filename + "_smoothed.png")
        plt.plot(vid_time, smoothed_data)
        plt.ylim(0, 90)
        plt.suptitle('Contact angle vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Contact angle (\u00B0)')
        plt.savefig(file_name, format="png", bbox_inches="tight", tranparent=True)
        print(f"Graph saved to {file_name}")

    if "butter" in lines:
        if combine:
            file_name = combined_graph
            plt.figure(0)
        else:
            plt.figure(4)
            file_name = out_folder_path.joinpath(out_filename + "_buttered.png")
        plt.plot(vid_time, butterfiltered)
        plt.ylim(0, 90)
        plt.suptitle('Contact angle vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Contact angle (\u00B0)')
        plt.savefig(file_name, format="png", bbox_inches="tight", tranparent=True)
        print(f"Graph saved to {file_name}")

    if "gust" in lines:
        if combine:
            plt.figure(0)
            file_name = combined_graph
        else:
            plt.figure(5)
            file_name = out_folder_path.joinpath(out_filename + "_gusted.png")
        plt.plot(vid_time, gustfiltered)
        plt.ylim(0, 90)
        plt.suptitle('Contact angle vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Contact angle (\u00B0)')
        plt.savefig(file_name, format="png", bbox_inches="tight", tranparent=True)
        print(f"Graph saved to {file_name}")


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


def process_video_from_path(path: Union[str, Path],
                            manual: bool = False
                            ) -> np.ndarray:

    if isinstance(path, str):
        path = Path(path)

    video_path = path.resolve()

    if video_path.exists():
        print(f"Video loaded from {video_path}")
        vid: videoType = cv2.VideoCapture(str(video_path))
    else:
        sys.exit("Error: Video file not found...")

    data_array = process_video(vid, manual=manual)
    return data_array


def process_videos_from_dirs(in_path: Union[str, Path], out_path: Union[str, Path] = "") -> None:
    """get all .avi files from folder and sub-folders as Paths and process each to out_path"""

    if isinstance(in_path, str):
        in_path = Path(in_path)

    if not out_path:
        out_path = in_path / "out"
    elif isinstance(out_path, str):
        out_path = Path(out_path)

    # get all .avi files from folder and sub-folders as Paths
    for video_path in in_path.glob('**/*.avi'):
        video_stem = video_path.stem
        raw_data = process_video_from_path(video_path)
        processed_data = process_videodata(raw_data)
        print_video_data(processed_data, out_path=data_path / 'out', out_filename=video_stem)


if __name__ == "__main__":
    data_path = Path(R'..\data')

    video_path = Path(R'..\data\thoria 300c dry 70.avi')
    video_stem = video_path.stem

    raw_data = process_video_from_path(video_path)
    processed_data = process_videodata(raw_data, flatten='low')
    print_video_data(processed_data, out_path=R'..\data\out', out_filename=video_stem, combine=True)
