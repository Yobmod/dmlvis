"""functions for image manipulation and processing"""
# -*- coding: utf-8 -*-
from __future__ import annotations
from datetime import datetime
import numpy as np
import cv2
import math
import imutils
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
# from functools import lru_cache
# import os
import sys
# import random
# import argparse

from typing import Tuple, Union, List, cast, overload  # , Any, NewType, TypeVar, Iterable,
from typing import Optional as Opt
from typing_extensions import Literal as Lit  # , Final as Fin
from mytypes import imageType, videoType, contourType, pointType, intArray, PILImage, colorType


def set_res(cap: cv2.VideoCapture, resolution: Union[int, str]) -> str:
    """."""
    if resolution in [480, "480", "480p"]:
        cap.set(3, 640)
        cap.set(4, 480)
    elif resolution in [1080, "1080", "1080p"]:
        cap.set(3, 1920)
        cap.set(4, 1080)
    elif resolution in [720, "720", "720p"]:
        cap.set(3, 1920)
        cap.set(4, 1080)
    else:
        resolution = 720
        set_res(cap, resolution)
    return str(resolution)

# @lru_cache(maxsize = 128, typed=True)


def get_outlined_image(frame: imageType) -> imageType:
    grayed: imageType = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred: imageType = cv2.GaussianBlur(grayed, (9, 9), 0)
    # cv2.imwrite(R'.\data\ceria_grey.png', grayed)
    # cv2.imwrite(R'.\data\ceria_blurred.png', blurred)
    # perform edge detection, then perform a dilation + erosion to close gaps
    edged: imageType = cv2.Canny(blurred, 20, 100)
    # cv2.imwrite(R'.\data\ceria_initedge.png', edged)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # cv2.imwrite(R'.\data\ceria_grey.png', grayed)
    return edged


# @lru_cache(maxsize=128, typed=True)
def get_largest_contour(frame: imageType) -> Opt[contourType]:
    result: Tuple[List[contourType], List[List[intArray]]]
    result = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[0]
    if len(contours) > 0:
        # if len(contours := result[0]) > 0:
        # grab the largest contour, then draw a mask for the pill l
        # cnt_areas = [cv2.contourArea(cntr) for cntr in contours]
        # c = max(cnt_areas)
        largest_contour: contourType = max(contours, key=cv2.contourArea)
    else:
        return None
        # raise ValueError("No contours identified in image")
    return largest_contour


# @lru_cache(maxsize=128, typed=True)
def get_contour_lims(frame: imageType) -> Tuple[int, int, int, int]:
    """get measurements from pixels in edged image, as tuple of (x_position, y_poition,
       width, height). If contour not found in image, returns tuple of zeros"""
    largest_contour = get_largest_contour(frame)
    if largest_contour is not None:
        (x, y, w, h) = cv2.boundingRect(largest_contour)
    else:
        (x, y, w, h) = (0, 0, 0, 0)
    return (x, y, w, h)


def calc_contact_angle(w: int, h: int) -> float:
    drop_h: float = h / 2
    if drop_h > 0:
        radius: float = (drop_h / 2) + ((w * w) / (8 * drop_h))
        opp = radius - drop_h
        hyp = radius
        sin_ca = opp / hyp
        ca: float = 90 - math.degrees(math.asin(sin_ca))
    else:
        radius = 0.0  # and opp = 0
        ca = 0
    print(f"drop_h={drop_h}, drop_w={w}, model_rad={radius}, angle={ca}")
    return ca


def drop_params(w: int, h: int) -> float:
    opp = h / 2
    ajd = w / 2
    tan_angle = opp / ajd
    angle = math.degrees(math.atan(tan_angle))
    return angle


# @lru_cache(maxsize=128, typed=True)
def get_image_skew(frame: imageType) -> float:
    """find contours in an edged image, calculate angle from horizontal of line bisecting
       left and right-most pixels. If contour not found in image, returns zero"""
    """find contours in an edged image, and calculate angle from horizontal"""
    largest_contour = get_largest_contour(frame)
    if largest_contour is not None:
        point: pointType
        # array of array of arrays of int, int -> list of tuple of int,int
        pxs_coords: List[pointType] = [cast(pointType, tuple(point_array[0])) for point_array in largest_contour if largest_contour is not None]
        x_coords: List[int] = [point[0] for point in pxs_coords]
        # y_coords: List[int] = [point[1] for point in pxs_coords]
        x_min: int = min(x_coords)
        x_max: int = max(x_coords)

        # get point with x_min
        x_min_points: List[Tuple[int, int]] = [point for point in pxs_coords if point[0] == x_min]
        x_max_points: List[Tuple[int, int]] = [point for point in pxs_coords if point[0] == x_max]
        y_of_max: float = sum(point[1] for point in x_max_points) / len(x_max_points)
        y_of_min: float = sum(point[1] for point in x_min_points) / len(x_min_points)

        # x_max_point = (x_max, y_of_max)
        # print(x_min_points, x_max_points)

        x_diff = x_max - x_min
        y_diff = y_of_max - y_of_min
        # skew is positive -> is clockwise
        skew = math.degrees(math.atan(y_diff / x_diff))
        # print(skew)
        # if y_of _max > y_of_min:
        # else: return -skew
    else:
        skew = 0.0
    return skew


# @lru_cache(maxsize=128, typed=True)
def crop_outlined_image(frame: imageType) -> Opt[imageType]:
    """find contours in an edged image, create a mask sized to largest contour and apply to image.
    If contour not found in image, return None"""
    largest_contour = get_largest_contour(frame)
    if largest_contour is not None:
        mask = np.zeros(frame.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=-1)  # color = opacity?

        # compute its bounding box of pill, then extract the ROI, and apply the mask
        h: int
        w: int
        x: int
        y: int
        (x, y, w, h) = cv2.boundingRect(largest_contour)
        imageROI = cast(imageType, frame[y:y + h, x:x + w])
        maskROI = mask[y:y + h, x:x + w]
        imageROI = cv2.bitwise_and(imageROI, imageROI, mask=maskROI)
        # skew = get_image_skew(frame)
        # if skew > 0:  # , need to rotateanticlockwise
        # imageROI = imutils.rotate_bound(imageROI, -skew)
        return imageROI
    else:
        return None


def save_image_groups(frames_list: List[imageType], save_folder: str = "data", raw: bool = True, edged: bool = False, masked: bool = False) -> None:
    # print(frames_list)
    today = datetime.today().strftime('%Y_%m_%d')
    now = datetime.now().strftime('%H%M%S')
    name = Path(f'./{save_folder}/images_{today}')
    name.mkdir(parents=True, exist_ok=True)

    if raw is True:
        for index, frame in enumerate(frames_list):
            cv2.imwrite(f'{name}/{index}_raw_{now}.png', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if edged is True or masked is True:
        edged_frames: List[imageType] = [get_outlined_image(frame) for frame in frames_list]

        if edged is True:
            for index, frame in enumerate(edged_frames):
                cv2.imwrite(f'{name}/{index}_edged_{now}.png', frame)

        if masked is True:
            masked_frames: List[imageType] = [crop_outlined_image(frame) for frame in edged_frames]
            for index, frame in enumerate(masked_frames):
                cv2.imwrite(f'{name}/{index}_mask_{now}.png', frame)


def add_image_text(image: imageType, text: str, underline: bool = False) -> imageType:
    """add text to image """

    text_image: imageType = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    h, w = text_image.shape[:2]
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_origin_pos = (int(w / 10), int(9 * h / 10))  # from topleft
    text_scale = 0.4
    text_color = (154, 205, 50)
    text_line_thick = 1
    cv2.putText(text_image, text, text_origin_pos, text_font, text_scale, text_color, text_line_thick, lineType=cv2.LINE_AA)

    if underline:
        under_start_offset = (-5, 10)  # -x to the left, y underneath
        line_length = 155
        under_start_pos = tuple(x + y for x, y in zip(text_origin_pos, under_start_offset))
        under_finish_pos = tuple(x + y for x, y in zip(under_start_pos, (line_length, 0)))
        under_color = (50, 100, 250)
        under_line_thick = 2
        cv2.line(text_image, under_start_pos, under_finish_pos, under_color, under_line_thick)

    return text_image


def annot_image(img: imageType, ang: float, txt_size: int = 10) -> PILImage:
    """convert opencv image array to PIL image, then annotate with contact angle"""

    pil_im_grey: PILImage = Image.fromarray(img)
    pil_im_color: PILImage = pil_im_grey.convert('RGB')  # needed to draw color text onto
    text_color: colorType = (255, 255, 0)
    text_position = (10, 10)
    text_size = txt_size
    texty = f"C. angle = {ang:.1f}"
    try:
        font: ImageFont.FreeTypeFont = ImageFont.truetype(R'/Library/Fonts/Arial.ttf', text_size)
    except IOError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(pil_im_color)
    draw.text(text_position, texty, font=font, fill=text_color)
    pil_im_color.save(R'.\data\ceria_annotated.bmp')
    return pil_im_color


def savitzky_golay(y: Union[List, 'np._ArrayLike[float]'], window_size: int, order: int, deriv: int = 0, rate: int = 1) -> np.ndarray:
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data. It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering approaches, such as moving averages techniques.

    Parameters
    ----------
    y : array_like, shape (N,) the values of the time history of the signal.
    window_size : int,  the length of the window. Must be an odd integer number.
    order : int the order of the polynomial used in the filtering.Must be less then `window_size` - 1.
    deriv: intthe order of the derivative to compute (default = 0 means only smoothing)

    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).

    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    print(y)
    print(y[0])
    # convert to array if list or tuple
    if isinstance(y, (list, tuple)):
        y = np.array(y)
        print(y)
        print(y[0])
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")

    if window_size < 1:
        raise TypeError("window_size size must be a positive odd number")

    if window_size % 2 != 1:
        window_size += 1
        print("window_size size must be a positive odd number, incremented by 1")

    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself

    win_begin: float = cast(float, y[0])
    win_end: float = cast(float, y[-1])
    firstvals = win_begin - np.abs(y[1:half_window + 1][::-1] - win_begin)
    lastvals = win_end + np.abs(y[-half_window - 1:-1][::-1] - win_end)
    y = np.concatenate((firstvals, y, lastvals))
    z: np.ndarray = np.convolve(m[::-1], y, mode='valid')
    return z


def zoom_image(img: imageType, zoom: float) -> imageType:
    # get the webcam image size
    height, width, channels = img.shape

    mask = np.zeros((height, width), dtype=np.uint8)
    new_top = int(height)
    new_bttm = int(height * 0.0)
    imageROI = cast(imageType, img[new_bttm:new_top, 0:width])
    maskROI = mask[new_bttm:new_top, 0:width]
    imageROI = cv2.bitwise_and(imageROI, imageROI, mask=maskROI)

    """
    # prepare the crop
    #(centerX, centerY) = (int(height / 2), int(width / 2))
    #if zoom:
        #(radiusX, radiusY) = int(zoom * height / 2), int(zoom * width / 2)
    #else:
        #(radiusX, radiusY) = (int(height / 2), int(width / 2))
    """

    # minX, maxX = centerX - radiusX, centerX + radiusX
    # minY, maxY = centerY - radiusY, centerY + radiusY
    minY = int(width * 0.3)
    maxY = int(width * 0.7)  # lower shift left
    minX = int(height * 0.5)
    maxX = int(height * 0.9)  # remove 0.2 and 0.4 = 0.6

    cropped = img[minX:maxX, minY:maxY]
    resized_cropped = cv2.resize(cropped, (width, height))

    return resized_cropped


def zoom_image_from_path(path: Union[Path, str], zoom: float) -> imageType: ...


def zoom_and_rotate_video(vid: videoType, zoom: float, rotate: float = 0, show: bool = True, save_name: str = "") -> videoType:
    """Zoom and rotate each frame of video, return new video array"""
    frame_no = 0
    out_folder_path = Path(f'../data/{save_name}')
    out_folder_path.mkdir(exist_ok=True)

    while vid.isOpened():  # and img.any():

        frame_out: Tuple[bool, imageType] = vid.read()
        # (ret, frame) = frame_out  ## loses typings :/
        ret = frame_out[0]
        frame = frame_out[1]

        if ret is True:

            if rotate:
                rotated = imutils.rotate(frame, rotate)
            else:
                rotated = frame

            cropped = zoom_image(rotated, zoom=zoom)

            if frame_no % 500 == 0:
                print(frame_no)
                cv2.imwrite(str(out_folder_path) + fR'\{save_name}_{frame_no}.png', cropped)
            frame_no += 1

            if show:
                cv2.imshow('Cropped window', cropped)

            key: int = cv2.waitKey(10) & 0xFF
            if key & 0xFF == ord("q"):  # quit camera if 'q' key is pressed
                break
        else:
            break  # break if problem with a frame
    else:
        print("Error: Video not opened...")

    vid.release()
    cv2.destroyAllWindows()


def load_video_from_path(path: Union[Path, str]) -> videoType:
    """Zoom and rotate video, save to out_path"""

    if isinstance(path, str):
        path = Path(path)
    video_path = path.resolve()
    if video_path.exists():
        print(f"Video loaded from {video_path}")
        vid: videoType = cv2.VideoCapture(str(video_path))
    else:
        sys.exit("Error: Video file not found...")

    return vid


def zoom_and_rotate_video_from_path(in_path: Union[Path, str], zoom: float, rotate: float = 0, save_name: str = "") -> None:
    """Zoom and rotate video, save to out_path"""
    if isinstance(in_path, str):
        in_path = Path(in_path)

    if not save_name:
        save_name = in_path.stem

    vid = load_video_from_path(in_path)
    zoom_and_rotate_video(vid, zoom=zoom, rotate=rotate, save_name=save_name)


def zoom_and_rotate_video_from_dirs(in_path: Union[Path, str], zoom: float, out_path: Union[str, Path] = "", rotate: float = 0) -> None:
    """get all .avi and mp4 files from folder and sub-folders as Paths and process each to out_path"""

    if isinstance(in_path, str):
        in_path = Path(in_path)

    if not out_path:
        out_path = in_path / "out"
    elif isinstance(out_path, str):
        out_path = Path(out_path)

    from itertools import chain
    # get all .avi files from folder and sub-folders as Paths
    for video_path in chain(in_path.glob('**/*.avi'), in_path.glob('**/*.mp4')):
        video_stem = video_path.stem
        zoom_and_rotate_video_from_path(video_path, zoom=zoom, rotate=rotate, save_name=video_stem)


def make_GIF(image_path: Union[Path, str]) -> None:
    """open all images in a dir and combine to a GIF."""
    import imageio
    from pygifsicle import optimize

    if isinstance(image_path, str):
        image_path = Path(image_path)

    image_dir = image_path.parent
    image_file = image_path.stem
    gif_path = image_dir / f"{image_file}.gif"
    gif_path = Path("./xxxx.gif")
    with imageio.get_writer(gif_path, mode='I') as writer:
        img_files = sorted((img_file for img_file in image_dir.glob('*.png')))
        for img_file in img_files:
            writer.append_data(imageio.imread(img_file))
    print(f"{len(img_files)} images loaded from {image_path}")
    try:
        optimize(gif_path)
    except Exception:
        print("gifsicle not installed")


@overload
def split_color_channels_from_array(img_array: 'np.ndarray[np.ndarray[int]]',
                                    channel: Lit['all']) -> Tuple[imageType, imageType, imageType]: ...


@overload
def split_color_channels_from_array(img_array: 'np.ndarray[np.ndarray[int]]',
                                    channel: Lit['red', 'blue', 'green']) -> imageType: ...


def split_color_channels_from_array(img_array: 'np.ndarray[np.ndarray[int]]',
                                    channel: Lit['red', 'blue', 'green', 'all'] = 'all'
                                    ) -> Union[imageType, Tuple[imageType, imageType, imageType]]:
    pil_img = Image.fromarray(img_array)
    (red, green, blue) = pil_img.split()
    if channel == 'red':
        return red
    elif channel == 'green':
        return green
    elif channel == 'blue':
        return blue
    else:
        return (red, green, blue)


@overload
def split_color_channels_from_path(image_path: Union[Path, str],
                                   channel: Lit['red', 'blue', 'green']) -> imageType: ...


@overload
def split_color_channels_from_path(image_path: Union[Path, str],
                                   channel: Lit['all']) -> Tuple[imageType, imageType, imageType]: ...


def split_color_channels_from_path(image_path: Union[Path, str],
                                   channel: Lit['red', 'blue', 'green', 'all'] = 'all'
                                   ) -> Union[imageType, Tuple[imageType, imageType, imageType]]:
    import imageio

    if isinstance(image_path, str):
        image_path = Path(image_path)

    out_path = image_path.parent
    name = image_path.stem

    img = imageio.imread(image_path)
    red, green, blue = split_color_channels_from_array(img, channel=channel)

    if channel == 'red':
        imageio.imwrite(out_path / f"{name}_red.jpg", red)
        return red
    elif channel == 'green':
        imageio.imwrite(out_path / f"{name}_red.jpg", green)
        return green
    elif channel == 'blue':
        imageio.imwrite(out_path / f"{name}_red.jpg", blue)
        return blue
    else:
        imageio.imwrite(out_path / f"{name}_red.jpg", red)
        imageio.imwrite(out_path / f"{name}_red.jpg", green)
        imageio.imwrite(out_path / f"{name}_red.jpg", blue)
        return (red, green, blue)


if __name__ == "__main__":
    # vid_path=Path("../data")
    # zoom_and_rotate_video_from_dirs(vid_path, zoom=1.0, rotate=90)

    # make_GIF(R"..\data\ceria 5%Europium 300c dry 40\ceria 5%Europium 300c dry 40_0.png")

    image_path = Path("../tests/test_images/test1.jpg")
    split_color_channels_from_path(image_path, channel='all')
