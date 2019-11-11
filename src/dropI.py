"""droplet recognition of single image"""
# -*- coding: utf-8 -*-
import cv2
from pathlib import Path

import imutils
from dmlutils import get_outlined_image, crop_outlined_image, get_contour_lims, calc_contact_angle, annot_image, get_image_skew

from typing import Type, NewType, Union, Optional, Tuple
from mytypes import imageType, colorType, PILImage


def process_image(img: imageType, name: str, save_images: bool = True, verbosity: int = 0) -> Tuple[float, float, float]:

    edged = get_outlined_image(img)
    rotated = imutils.rotate_bound(edged, -90)
    skew = get_image_skew(rotated)
    fixed = imutils.rotate_bound(rotated, -skew)
    (x, y, w, h) = get_contour_lims(fixed)
    angle: float = calc_contact_angle(w, h)
    w, h, angle = (round(w, 2), round(h, 2), round(angle, 2))
    if save_images:
        cv2.imwrite(Rf'..\data\{name}_edged.png', edged)
        mask = crop_outlined_image(edged)
        cv2.imwrite(Rf'..\data\{name}_masked.png', mask)
        annot_image(mask, angle, txt_size=10, save_path=Rf'..\data\{name}_annotated.bmp')
        # annotated = add_image_text(mask, f"C. angle = {ang:.1f}")
        # cv2.imwrite(R'.\data\test_annotated.png', annotated)
    if verbosity > 0:
        print("image processing done")
    return (w, h, angle)


def process_image_from_path(path: Union[str, Path], name: str = None) -> Optional[Tuple[float, float, float]]:
    """tries to load image from path. If successful, processes and saves images, and returns width, height, contact angle as Tuple"""
    if name is not None:
        name_from_path = name
    else:
        path_fix = Path(path)
        name_from_path = path_fix.stem

    img_from_path: imageType = cv2.imread(path, cv2.IMREAD_COLOR)

    if img_from_path is not None:
        print("image loaded")
        angle = process_image(img_from_path, name_from_path)
        return angle
    else:
        print("Image not loaded...")
        return None


def display_image(img: imageType) -> None:
    while img is not None:

        cv2.imshow('test image', img)

        key: int = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord("q"):         # wait for ESC key or Q to exit
            break
    cv2.destroyAllWindows()


def display_image_from_path(path: Union[str, Path]) -> Optional[imageType]:
    """tries to load image from path. If successful, opens image in window and saves images, and returns original image"""
    img_from_path: imageType = cv2.imread(path, cv2.IMREAD_COLOR)

    if img_from_path is not None:
        print("image loaded")
        display_image(img_from_path)
        return img_from_path
    else:
        print("Image not loaded...")
        return None
