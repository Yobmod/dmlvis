import numpy as np

lhi = [1, 4, 6, 3, 8, 90, 101, 56, 32]
arr = np.array(lhi, dtype=float)

for i, x in enumerate(arr):
    if x > 80:
        print(x)
        arr[i] = np.NaN

print(arr)

"""
for x in arr:
    if x > 80:
        arr[x] = np.NaN
"""

import cv2
import imageio
from pathlib import Path
from PIL import Image
from typing import Union, Tuple
from typing_extensions import Literal
from mytypes import imageType




