from typing import Union
from pathlib import Path

from PIL import Image


def verifyTruncated(path: Union[str, Path]) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
            if image.format is None:
                return False
            return True
    except (IOError, OSError):
        return False
