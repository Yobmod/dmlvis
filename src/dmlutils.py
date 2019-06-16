"""."""
# -*- coding: utf-8 -*-
from datetime import datetime
import numpy as np
import cv2
import math
import imutils
import pathlib
from PIL import Image, ImageFont, ImageDraw
from functools import lru_cache

from typing import Tuple, Union, List, Iterable, cast  # , Any, NewType, TypeVar
from typing import Optional as Opt
from mytypes import imageType, contourType, pointType, intArray, PILImage, colorType


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
    #cv2.imwrite(R'.\data\ceria_grey.png', grayed)
    #cv2.imwrite(R'.\data\ceria_blurred.png', blurred)
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
    "get left and right most pixels in edged image"
    largest_contour = get_largest_contour(frame)
    (x, y, w, h) = cv2.boundingRect(largest_contour)
    return (x, y, w, h)


def calc_contact_angle(w: int, h: int) -> float:
    drop_h: float = h / 2
    radius: float = (drop_h / 2) + ((w * w) / (8 * drop_h))
    opp = radius - drop_h
    hyp = radius
    sin_ca = opp / hyp
    ca: float = 90 - math.degrees(math.asin(sin_ca))
    # print(f"drop_h={drop_h}, drop_w={w}, model_rad={radius}, angle={ca}")
    return ca


def drop_params(w: int, h: int) -> float:
    opp = h / 2
    ajd = w / 2
    tan_angle = opp / ajd
    angle = math.degrees(math.atan(tan_angle))
    return angle


# @lru_cache(maxsize=128, typed=True)
def get_image_skew(frame: imageType) -> float:
    """find contours in an edged image, create a mask sized to largest contour and apply to image"""
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
        y_of_max: float = sum([point[1] for point in x_max_points]) / len(x_max_points)
        y_of_min: float = sum([point[1] for point in x_min_points]) / len(x_min_points)

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
def crop_outlined_image(frame: imageType) -> imageType:
    """find contours in an edged image, create a mask sized to largest contour and apply to image"""
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
        return frame


def save_image_groups(frames_list: List[imageType], save_folder: str = "data", raw: bool = True, edged: bool = False, masked: bool = False) -> None:
    # print(frames_list)
    today = datetime.today().strftime('%Y_%m_%d')
    now = datetime.now().strftime('%H%M%S')
    name = pathlib.Path(f'./{save_folder}/images_{today}')
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


def annot_image(img: imageType, ang: float, txt_size: int = 10) -> None:
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
