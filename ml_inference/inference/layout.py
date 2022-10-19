from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import layoutparser as lp
import numpy as np
from PIL import Image

from ml_inference.logger import get_logger
import ml_inference.models.tesseract as tesseract
import ml_inference.models.detectron2 as detectron2

logger = get_logger()


@dataclass
class LayoutElement:
    type: str
    # NOTE(robinson) - Coordinates are reported starting from the upper left and
    # proceeding clockwise
    coordinates: List[Tuple[float, float]]
    text: Optional[str] = None

    def __str__(self):
        return self.text

    def to_dict(self):
        return self.__dict__


class DocumentLayout:
    """Class for handling documents that are saved as .pdf files. For .pdf files, a
    document image analysis (DIA) model detects the layout of the page prior to extracting
    element."""

    def __init__(self):
        self._pages = None

    def __str__(self) -> str:
        return "\n\n".join([str(page) for page in self.pages])

    @property
    def pages(self) -> List[PageLayout]:
        """Gets all elements from pages in sequential order."""
        return self._pages

    @classmethod
    def from_pages(cls, pages: List[PageLayout]) -> DocumentLayout:
        """Generates a new instance of the class from a list of `PageLayouts`s"""
        doc_layout = cls()
        doc_layout._pages = pages
        return doc_layout

    @classmethod
    def from_file(cls, filename: str):
        logger.info(f"Reading PDF for file: {filename} ...")
        layouts, images = lp.load_pdf(filename, load_images=True)
        pages: List[PageLayout] = list()
        for i, layout in enumerate(layouts):
            image = images[i]
            # NOTE(robinson) - In the future, maybe we detect the page number and default
            # to the index if it is not detected
            page = PageLayout(number=i, image=image, layout=layout)
            page.get_elements()
            pages.append(page)
        return cls.from_pages(pages)


class PageLayout:
    """Class for an individual PDF page."""

    def __init__(self, number: int, image: Image, layout: lp.Layout):
        self.image = image
        self.image_array: Union[np.ndarray, None] = None
        self.layout = layout
        self.number = number
        self.elements: List[LayoutElement] = list()

    def __str__(self):
        return "\n\n".join([str(element) for element in self.elements])

    def get_elements(self, inplace=True) -> Optional[List[LayoutElement]]:
        """Uses a layoutparser model to detect the elements on the page."""
        logger.info("Detecting page elements ...")
        detectron2.load_model()

        elements = list()
        # NOTE(mrobinson) - We'll want make this model inference step some kind of
        # remote call in the future.
        image_layout = detectron2.model.detect(self.image)
        # NOTE(robinson) - This orders the page from top to bottom. We'll need more
        # sophisticated ordering logic for more complicated layouts.
        image_layout.sort(key=lambda element: element.coordinates[1], inplace=True)
        for item in image_layout:
            text_blocks = self.layout.filter_by(item, center=True)
            text = str()
            for text_block in text_blocks:
                # NOTE(robinson) - If the text attribute is None, that means the PDF isn't
                # already OCR'd and we have to send the snippet out for OCRing.
                if text_block.text is None:
                    text_block.text = self.ocr(text_block)
            text = " ".join([x for x in text_blocks.get_texts() if x])

            elements.append(
                LayoutElement(type=item.type, text=text, coordinates=item.points.tolist())
            )

        if inplace:
            self.elements = elements
            return None
        return elements

    def ocr(self, text_block: lp.TextBlock) -> str:
        """Runs a cropped text block image through and OCR agent."""
        logger.debug("Running OCR on text block ...")
        tesseract.load_agent()
        image_array = self._get_image_array()
        padded_block = text_block.pad(left=5, right=5, top=5, bottom=5)
        cropped_image = padded_block.crop_image(image_array)
        return tesseract.ocr_agent.detect(cropped_image)

    def _get_image_array(self) -> Union[np.ndarray, None]:
        """Converts the raw image into a numpy array."""
        if self.image_array is None:
            self.image_array = np.array(self.image)
        return self.image_array
