
from typing import List, Dict, Union
from pathlib import Path
from mytypes import pathType


class Ion():
    def __init__(self, element: str, energy: float) -> None:
        ...


class Layer():

    def __init__(self, attrib: Dict[str, Dict[str, float]], density: float, width: float, name: str = ..., phase: int=...) -> None:
        ...


class Target():
    def __init__(self, layers: List[Layer]) -> None:
        ...


class TRIM():
    def __init__(self, target: Target, ion: Ion, number_ions: int, calculation: int = 1) -> None:
        ...

    def run(self, dir: Union[str, Path]) -> str:
        ...

    @staticmethod
    def copy_output_files(dir: pathType, out: pathType) -> None:
        ...
