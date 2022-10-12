import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.controllers import health_check_controller, detection_controller

app = FastAPI(version='1.0', title='VIP-VIRA',
              description="API for processing images")

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

app.include_router(
    health_check_controller.router,
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    detection_controller.router,
    prefix="/Inference",
    tags=["Inference"],
    responses={404: {"description": "Not found"}},
)
