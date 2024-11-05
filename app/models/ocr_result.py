from pydantic import BaseModel
from typing import List, Tuple

class BoundingBox(BaseModel):
    top_left: Tuple[float, float]
    top_right: Tuple[float, float]
    bottom_right: Tuple[float, float]
    bottom_left: Tuple[float, float]

class TextDetection(BaseModel):
    text: str
    confidence: float
    bounding_box: BoundingBox

class OCRResult(BaseModel):
    results: List[TextDetection]

    @classmethod
    def from_easyocr_result(cls, result: List[Tuple[List[Tuple[int, int]], str, float]]) -> 'OCRResult':
        text_results: List[TextDetection] = []
        for detection in result:
            bounding_box = BoundingBox(
                top_left=detection[0][0],
                top_right=detection[0][1],
                bottom_right=detection[0][2],
                bottom_left=detection[0][3]
            )
            text_results.append(TextDetection(
                text=detection[1],
                confidence=float(detection[2]),
                bounding_box=bounding_box
            ))
        return cls(results=text_results)