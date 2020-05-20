from typing import Type, TYPE_CHECKING, Tuple, Union, List, NewType  # , Any, NewType, TypeVar
from PIL import Image
import numpy as np
from typing_extensions import Literal

if TYPE_CHECKING:  # for mypy
    class ndarray(np.ndarray): ...
    imageType = NewType('imageType', np._ArrayLike[np.ndarray[np.ndarray[int]]])
    videoType = NewType('videoType', np._ArrayLike[imageType])
    contourType = NewType('contourType', np._ArrayLike[np.ndarray[np.ndarray[int]]])
    intArray = np.ndarray[int]
    floatArray = np.ndarray[float]
else:  # python doesnt allow ndarray subscriptable yet
    imageType = np.ndarray
    videoType = np.ndarray
    contourType = np.ndarray
    intArray = np.ndarray
    floatArray = np.ndarray


class PILImage(Image.Image): ...


colorHSV = Tuple[float, float, float]
colorRGB = Tuple[int, int, int]
colorType = Union[colorHSV, colorRGB]

RGBLit = Literal["RGB", "rgb"]
HSVLit = Literal["HSV", "hsv"]
colorSystemLit = Union[RGBLit, HSVLit]


pointType = Tuple[int, int]
