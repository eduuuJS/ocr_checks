from fastapi import APIRouter, File, UploadFile, HTTPException

from app.models.ocr_result import OCRResult
from app.services.check_services import CheckServices

router: APIRouter = APIRouter()
check_service: CheckServices = CheckServices()


@router.post("/process/checks")
async def process_checks(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, detail="File uploaded is not an image.")

    results: OCRResult = await check_service.process_checks(file)

    return results

