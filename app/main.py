from fastapi import FastAPI
from app.controllers.image_controller import router as image_router
from app.controllers.checks_controller import router as checks_router

app: FastAPI = FastAPI()

app.include_router(image_router)
app.include_router(checks_router)

@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "App is running"}
