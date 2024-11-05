import cv2
import numpy as np
import easyocr
from fastapi import UploadFile

from app.models.labels_definition import LabelsDefinition
from app.utils.image_processing import preprocess_image
from app.models.ocr_result import OCRResult, BoundingBox, TextDetection
from typing import List, Tuple, Optional

from app.utils.label_comparation import label_comparation

def find_best_match(result: List[TextDetection], label: str, similarity_threshold: float = 70) -> Optional[TextDetection]:
    max_similarity: float = 0
    best_match: Optional[TextDetection] = None

    for detection in result:
        similarity: float = label_comparation(label, detection.text)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = detection

    if max_similarity > similarity_threshold:
        return best_match
    return None

def calculate_bottom_box(bounding_box: BoundingBox) -> float:
    return (bounding_box.bottom_right[1] + bounding_box.bottom_left[1]) / 2

def find_candidates(bottom_label: float, result: List[TextDetection], tolerance: float = 50) -> List[TextDetection]:
    candidates: List[TextDetection] = []

    for detection in result:
        top_label: float = (detection.bounding_box.top_left[1] + detection.bounding_box.top_right[1]) / 2
        minus_tolerance: float = tolerance * -1
        if minus_tolerance <= top_label - bottom_label <= tolerance:
            candidates.append(detection)
    return candidates

def find_candidates_unidirectional(bottom_label: float, result: List[TextDetection], low_threshold: float = 180, high_threshold: float = 320) -> List[TextDetection]:
    candidates: List[TextDetection] = []

    for detection in result:
        top_label: float = (detection.bounding_box.top_left[1] + detection.bounding_box.top_right[1]) / 2
        if low_threshold <=top_label - bottom_label <= high_threshold:
            candidates.append(detection)
    return candidates

def find_best_candidate(bounding_box: BoundingBox, candidates: List[TextDetection]) -> Optional[TextDetection]:
    right_label: float = (bounding_box.top_right[0] + bounding_box.bottom_right[0]) / 2
    min_distance: float = 10000
    best_candidate: Optional[TextDetection] = None

    for candidate in candidates:
        right_candidate: float = (candidate.bounding_box.top_right[0] + candidate.bounding_box.bottom_right[0]) / 2
        distance: float = abs(right_candidate - right_label)
        if distance < min_distance:
            min_distance = distance
            best_candidate = candidate

    return best_candidate

def find_account_number(result: List[TextDetection]) -> str:

    best_match: Optional[TextDetection] = find_best_match(result, LabelsDefinition.ACCOUNT_NUMBER.value)
    if best_match is None:
        return ""

    candidates: List[TextDetection] = find_candidates(calculate_bottom_box(best_match.bounding_box), result)

    best_candidate: Optional[TextDetection] = find_best_candidate(best_match.bounding_box, candidates)

    if best_candidate is None:
        return ""

    return best_candidate.text

def find_serie(result:List[TextDetection]) -> str:

    best_match: Optional[TextDetection] = find_best_match(result, LabelsDefinition.SERIE.value)
    if best_match is None:
        return ""

    candidates: List[TextDetection] = find_candidates(calculate_bottom_box(best_match.bounding_box), result)

    best_candidate: Optional[TextDetection] = find_best_candidate(best_match.bounding_box, candidates)

    if best_candidate is None or len(best_candidate.text) > 3:
        return ""

    return best_candidate.text

def find_check_number(result:List[TextDetection]) -> str:

    best_match: Optional[TextDetection] = find_best_match(result, LabelsDefinition.CHECK_NUMBER.value)
    if best_match is None:
        return ""

    candidates: List[TextDetection] = find_candidates(calculate_bottom_box(best_match.bounding_box), result)

    best_candidate: Optional[TextDetection] = find_best_candidate(best_match.bounding_box, candidates)

    if best_candidate is None:
        return ""

    return best_candidate.text

def find_pivote(result:List[TextDetection]) -> TextDetection:
    best_match_diferido: Optional[TextDetection] = find_best_match(result, LabelsDefinition.CHECK_TITLE_DIFERIDO.value, 90)
    best_match_vista: Optional[TextDetection] = find_best_match(result, LabelsDefinition.CHECK_TITLE_VISTA.value, 90)

    pivote: TextDetection = best_match_diferido if best_match_diferido is not None else best_match_vista
    return pivote

def find_footprints(result: List[TextDetection], pivote: TextDetection) -> List[TextDetection]:
    candidates: List[TextDetection] = find_candidates(calculate_bottom_box(pivote.bounding_box), result, tolerance=70)
    pivote_left: float = (pivote.bounding_box.top_left[0] + pivote.bounding_box.bottom_left[0]) / 2

    candidates_with_position: List[Tuple[TextDetection, float]] = []
    for candidate in candidates:
        box_left: float = (candidate.bounding_box.top_left[0] + candidate.bounding_box.bottom_left[0]) / 2
        if box_left < pivote_left:
            candidates_with_position.append((candidate, box_left))

    candidates_with_position.sort(key=lambda x: x[1])
    ordered_candidates: List[TextDetection] = [candidate for candidate, _ in candidates_with_position]

    return ordered_candidates

def format_footprints(footprints: List[TextDetection]) -> List[str]:
    ordered_texts: List[str] = [footprint.text for footprint in footprints]
    ordered_texts[0] = ordered_texts[0].replace("0", "O").upper()
    for i in range(1, len(ordered_texts)):
        if i != 0:
            ordered_texts[i] = ordered_texts[i].upper().replace("O", "0").replace("J", "1").upper()

    return ordered_texts

def calculate_footprints_bottom(footprints: List[TextDetection]) -> float:
    bottom: float = 0
    for footprint in footprints:
        footprint_bottom: float = (footprint.bounding_box.bottom_right[1] + footprint.bounding_box.bottom_left[1]) / 2
        bottom = bottom + footprint_bottom
    return bottom/len(footprints)

def find_suc_entity(result: List[TextDetection], foot_print_bottom: float) -> Optional[TextDetection]:
    candidates: List[TextDetection] = find_candidates(foot_print_bottom, result, tolerance=120)

    min_distance: float = 100000
    best_candidate: Optional[TextDetection] = None

    for candidate in candidates:
        left_candidate: float = (candidate.bounding_box.top_left[0] + candidate.bounding_box.bottom_left[0]) / 2
        if left_candidate < min_distance:
            min_distance = left_candidate
            best_candidate = candidate

    return best_candidate

def find_address_entity(result: List[TextDetection], suc_entity_bottom: float) -> Optional[TextDetection]:
    candidates: List[TextDetection] = find_candidates(suc_entity_bottom, result, tolerance=70)

    min_distance: float = 100000
    best_candidate: Optional[TextDetection] = None

    for candidate in candidates:
        left_candidate: float = (candidate.bounding_box.top_left[0] + candidate.bounding_box.bottom_left[0]) / 2
        if left_candidate < min_distance:
            min_distance = left_candidate
            best_candidate = candidate

    return best_candidate

def find_client_name(result: List[TextDetection]) -> Optional[TextDetection]:
    best_match: Optional[TextDetection] = find_best_match(result, LabelsDefinition.CHEK_SUM.value)

    print(best_match)

    if best_match is None:
        return None

    candidates: List[TextDetection] = find_candidates_unidirectional(calculate_bottom_box(best_match.bounding_box), result)
    normalized_candidates: List[TextDetection] = []
    for candidate in candidates:
        if candidate.confidence > 0.35:
            normalized_candidates.append(candidate)

    best_candidate: Optional[TextDetection] = None
    min_distance: float = 10000
    for candidate in normalized_candidates:
        distance: float = (candidate.bounding_box.top_left[1] + candidate.bounding_box.top_right[1]) / 2
        if distance < min_distance:
            min_distance = distance
            best_candidate = candidate

    return best_candidate

def find_client_address(result: List[TextDetection], client_name_box: BoundingBox) -> Optional[TextDetection]:
    candidates: List[TextDetection] = find_candidates(calculate_bottom_box(client_name_box), result, tolerance=70)

    best_candidate : Optional[TextDetection] = find_best_candidate(client_name_box, candidates)

    return best_candidate

def find_client_document(result: List[TextDetection], client_address_box: BoundingBox) -> Optional[TextDetection]:
    candidates: List[TextDetection] = find_candidates(calculate_bottom_box(client_address_box), result, tolerance=70)

    best_candidate : Optional[TextDetection] = find_best_candidate(client_address_box, candidates)

    return best_candidate


class CheckServices:
    def __init__(self):
        self.reader: easyocr.Reader = easyocr.Reader(['es'])

    async def process_checks(self, file: UploadFile) -> dict[str, any]:
        contents: bytes = await file.read()
        nparr: np.ndarray = np.frombuffer(contents, np.uint8)
        image: np.ndarray = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        image: np.ndarray = preprocess_image(image, self.reader)

        result: List[Tuple[List[Tuple[int, int]], str, float]] = self.reader.readtext(image)

        result_formated: OCRResult = OCRResult.from_easyocr_result(result)

        print("Finalizado OCR")

        account_number: str = find_account_number(result_formated.results)
        serie: str = find_serie(result_formated.results)
        check_number: str = find_check_number(result_formated.results)
        pivote: TextDetection = find_pivote(result_formated.results)
        type_check: str = pivote.text
        footprints: List[TextDetection] = find_footprints(result_formated.results, pivote)
        footprints_formated: List[str] = format_footprints(footprints)
        footprints_bottom: float = calculate_footprints_bottom(footprints)
        suc_entity: Optional[TextDetection] = find_suc_entity(result_formated.results, footprints_bottom)
        address_entity: Optional[TextDetection] = find_address_entity(result_formated.results, calculate_bottom_box(suc_entity.bounding_box))
        client_name: Optional[TextDetection] = find_client_name(result_formated.results)
        client_address: Optional[TextDetection] = None
        if client_name is not None:
            client_address: Optional[TextDetection] = find_client_address(result_formated.results, client_name.bounding_box)
        client_document: Optional[TextDetection] = None
        if client_address is not None:
            client_document: Optional[TextDetection] = find_client_document(result_formated.results, client_address.bounding_box)


        return {
            "account_number": account_number,
            "serie": serie,
            "check_number": check_number,
            "type_check": type_check,
            "footprints": footprints_formated,
            "suc_entity": suc_entity.text if suc_entity is not None else None,
            "address_entity": address_entity.text if address_entity is not None else None,
            "client_name": client_name.text if client_name is not None else None,
            "client_address": client_address.text if client_address is not None else None,
            "client_document": client_document.text if client_document is not None else None,
        }

