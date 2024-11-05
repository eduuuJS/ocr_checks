import cv2
import numpy as np
import easyocr
from fastapi import UploadFile
from app.utils.image_processing import preprocess_image
from app.models.ocr_result import OCRResult
from typing import List, Tuple

class OCRService:
    def __init__(self):
        self.reader: easyocr.Reader = easyocr.Reader(['es'])

    async def process_image(self, file: UploadFile) -> OCRResult:
        contents: bytes = await file.read()
        nparr: np.ndarray = np.frombuffer(contents, np.uint8)
        image: np.ndarray = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        image: np.ndarray = preprocess_image(image, self.reader)

        result: List[Tuple[List[Tuple[int, int]], str, float]] = self.reader.readtext(image)

        return OCRResult.from_easyocr_result(result)