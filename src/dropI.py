"""droplet recognition of single image"""
# -*- coding: utf-8 -*-
import cv2
from pathlib import Path

from dmlutils import get_outlined_image, crop_outlined_image, get_contour_lims, calc_contact_angle, annot_image  # set_res

from typing import Type, NewType, Union, Optional
from mytypes import imageType, colorType, PILImage


def process_image(img: imageType, name: str) -> None:

    edged = get_outlined_image(img)
    mask = crop_outlined_image(edged)
    cv2.imwrite(Rf'..\data\{name}_edged.png', edged)
    cv2.imwrite(Rf'..\data\{name}_masked.png', mask)
    (x, y, w, h) = get_contour_lims(edged)
    ang = calc_contact_angle(w, h)
    # annotated = add_image_text(mask, f"C. angle = {ang:.1f}")
    # cv2.imwrite(R'.\data\test_annotated.png', annotated)
    annot_image(mask, ang, txt_size=10, save_path=Rf'..\data\{name}_annotated.bmp')
    print("image processing done")


def process_image_from_path(path: Union[str, Path], name: str = None) -> Optional[imageType]:
    """tries to load image from path. If successful, processes and saves images, and returns original image"""
    if name is not None:
        name_from_path = name
    else:
        path_fix = Path(path)
        name_from_path = path_fix.stem

    img_from_path: imageType = cv2.imread(path, cv2.IMREAD_COLOR)

    if img_from_path is not None:
        print("image loaded")
        process_image(img_from_path, name_from_path)
        return img_from_path
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


process_image_from_path(R'..\data\ceria.png', "ce2")
display_image_from_path(R'..\data\ceria.png')
