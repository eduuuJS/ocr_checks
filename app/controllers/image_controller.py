from __future__ import annotations

from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.ocr_services import OCRService
from app.models.ocr_result import OCRResult

router: APIRouter = APIRouter()
ocr_service: OCRService = OCRService()

@router.post("/uploadImage/")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, detail="File uploaded is not an image.")

    results: OCRResult = await ocr_service.process_image(file)

    return results