import io
from pathlib import Path
from typing import Union

from PIL import Image


class GIFMaker:
    def __init__(self):
        self.frames: list[Union[Path, io.BytesIO, Image.Image]] = []

    def append_img(self, img: Union[Path, io.BytesIO, Image.Image]):
        self.frames.append(img)

    def next_img_file(self) -> io.BytesIO:
        new_fileobj = io.BytesIO()
        self.frames.append(new_fileobj)
        return new_fileobj

    def make_gif(
        self, output_file: Path, duration: int, loop: int = 0, optimize: bool = True
    ):
        _frames: list[Image.Image] = []
        for frame in self.frames:
            if not isinstance(frame, Image.Image):
                frame = Image.open(frame)
            _frames.append(frame)
        _frames[0].save(
            output_file,
            format="GIF",
            append_images=_frames,
            save_all=True,
            duration=duration,
            loop=loop,
            optimize=optimize,
        )

    def clear(self):
        self.frames.clear()
