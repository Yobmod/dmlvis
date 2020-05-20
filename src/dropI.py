"""."""
# -*- coding: utf-8 -*-
import cv2


from dmlutils import get_outlined_image, crop_outlined_image, get_contour_lims, calc_contact_angle, annot_image  # set_res

from typing import Type, NewType
from mytypes import imageType, colorType, PILImage


# raw_frames: t.List[imageType] = []

img: imageType = cv2.imread(R'.\data\ceria.png', cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE  /  cv2.IMREAD_UNCHANGED

while img is not None:  # and img.any():

    cv2.imshow('test image', img)

    key: int = cv2.waitKey(0) & 0xFF

    if key == ord('p'):  # wait for 'p' key to process

        edged = get_outlined_image(img)
        mask = crop_outlined_image(edged)
        cv2.imwrite(R'.\data\ceria_edged.png', edged)
        cv2.imwrite(R'.\data\ceria_masked.png', mask)
        (x, y, w, h) = get_contour_lims(edged)
        if w > 0 and h > 0:
            ang = calc_contact_angle(w, h)
        # annotated = add_image_text(mask, f"C. angle = {ang:.1f}")
        # cv2.imwrite(R'.\data\test_annotated.png', annotated)
            annot_image(mask, ang, txt_size=10)
        print("image processing done") 

    elif key == 27 or key == ord("q"):         # wait for ESC key or Q to exit
        break
else:
    print("Image not loaded...")

cv2.destroyAllWindows()

# gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 0=vert, 1=horiz, -1=both       # Display the resulting frame
# vflipped: imageType = cv2.flip(frame, 0)
# 0=vert, 1=horiz, -1=both       # Display the resulting frame
# hflipped: imageType = cv2.flip(frame, 1)

# blurred = cv2.GaussianBlur(frame, (3, 3), 0)
# # edged = cv2.Canny(gray, 20, 100)
"""
save_image_groups(raw_frames)
"""
