from __future__ import annotations
import os
from skimage.color import rgb2lab, deltaE_cie76
import numpy as np
import cv2
import imageio
from pathlib import Path
from PIL import Image
from ordered_set import OrderedSet
import colorsys
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# from plantcv import plantcv as pcv

from typing import Union, Tuple, List, cast, Set, Counter, Optional, overload
from typing_extensions import Literal
from mytypes import imageType, colorType, colorRGB, colorHSV, RGBLit, HSVLit, colorSystemLit


def get_image_from_path(path: Union[str, Path]) -> imageType:
    """retrun RGB image array if path is correct, else raise error"""
    try:
        image_BGR = cv2.imread(str(path))
        image: imageType = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
    except Exception:
        raise
    # wrong path results in None. Also check array is not empty
    if image is not None and image.any():
        return image
    else:
        raise FileNotFoundError


@overload
def get_image_color_list(image: imageType, system: RGBLit, number: int = ...) -> List[colorRGB]: ...


@overload
def get_image_color_list(image: imageType, system: HSVLit, number: int = ...) -> List[colorHSV]: ...


def get_image_color_list(image: imageType, system: colorSystemLit, number: int = 0) -> Union[List[colorHSV], List[colorRGB]]:  # List[colorType]:
    """Takes RGB image, get list of all colors in image, and returns as list of RGB or HSV tuples"""
    color_list: List[colorType]
    color: colorType

    if any(cs in system for cs in ("RGB", "rgb")):
        color_list = [cast(colorRGB, tuple(color)) for row in image for color in row]
        color_list.sort(key=lambda rgb: colorsys.rgb_to_hsv(*rgb))
    elif any(cs in system for cs in ("HSV", "hsv")):
        color_list = [cast(colorHSV, colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)) for row in image for r, g, b in row]  # type: ignore
        color_list.sort()
    else:
        raise ValueError("Please specifiy color system ('RGB or HSV')")

    if number:  # is not 0
        color_list = color_list[0:number]

    # print(len(color_list))
    return color_list


@overload
def get_image_color_set(image: imageType, system: RGBLit, number: int = ...) -> OrderedSet[colorRGB]: ...


@overload
def get_image_color_set(image: imageType, system: HSVLit, number: int = ...) -> OrderedSet[colorHSV]: ...


def get_image_color_set(image: imageType,
                        system: colorSystemLit,
                        number: int = 0
                        ) -> Union[OrderedSet[colorRGB], OrderedSet[colorHSV]]:
    """Takes RGB image, get set of all colors in an image, return as set of RGV or HSV tuples"""
    color_set: OrderedSet[colorType]

    color_list = get_image_color_list(image=image, system=system)  # don't pass number
    color_set = OrderedSet(color_list)  # list(dict.fromkeys(color_list_all)) to preserve order

    if number:  # is not 0
        color_set = color_set[:number]
    return color_set


@overload
def get_image_color_counts(image: imageType, system: RGBLit, number: int = 0) -> Counter[colorRGB]: ...


@overload
def get_image_color_counts(image: imageType, system: HSVLit, number: int = 0) -> Counter[colorHSV]: ...


def get_image_color_counts(image: imageType, system: colorSystemLit, number: int = 0) -> Union[Counter[colorHSV], Counter[colorRGB]]:
    """Takes RGB image, get counts of all colors in image, return as counter of RGB or HSV tuples"""
    # color_count: Counter[colorType]
    color_list = get_image_color_list(image=image, system=system)  # don't pass number
    color_count = Counter(color_list)
    if number:
        color_count = Counter(dict(color_count.most_common(number)))  # List[Tuple[Tuple[int, int, int], int]]
    return color_count


def plot_hue_histogram(data: Counter[colorHSV]) -> None:
    # hues: List[int] = [int(key[0] * 360) for key in data.keys()]
    # counts: List[int] = [count for count in data.values()]
    data_list = [(hsv[0] * 360, count) for (hsv, count) in data.items()]
    print(data_list)


def RGB2HEX(color: colorRGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_color_cluster_count_from_image(image: imageType, number_colors: int) -> None:

    plt.imshow(image)
    reshaped_image = image.reshape((image.shape[0] * image.shape[1], 3))
    clf = KMeans(n_clusters=number_colors)
    labels = clf.fit_predict(reshaped_image)
    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[key] for key in counts.keys()]

    rgb_colors = [ordered_colors[key] for key in counts.keys()]
    rgb_colors.sort(key=lambda rgb: colorsys.rgb_to_hsv(*rgb))
    hex_colors = [RGB2HEX(color) for color in rgb_colors]
    # hsv_colors = [colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)  for r, g, b in rgb_colors]  # type: ignore

    plt.figure(figsize=(8, 6))
    plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
    plt.show()


if __name__ == "__main__":

    # blue_str = Path(R"../tests/test_images/blues.jpg")
    # img = get_image_from_path(blue_str)
    # colors = get_image_color_set(img, "HSV", number=100)
    # cc = get_image_color_counts(img, "hsv", number=0)
    # plot_hue_histogram(cc)
    # color_histogram = pcv.analyze_color(rgb_img=img, mask=None, hist_plot_type='rgb')
    # color_histogram.save(Path(R"../tests/test_images/blueshist2.jpg"))

    # color_histogram_mpl = color_histogram.draw()
    # color_histogram_mpl.savefig(Path(R"../tests/test_images/blueshist.jpg"))

    fluor_path = Path(R"../tests/test_images/fl2.png")
    img = get_image_from_path(fluor_path)
    get_color_cluster_count_from_image(img, 20)
