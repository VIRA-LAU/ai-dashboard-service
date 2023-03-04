from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.controllers import health_check_controller, detection_controller

tags_metadata = [
    {
        "name": "Health",
        "description": "Checks the functioning of the endpoint.",
    },
    {
        "name": "Inference",
        "description": "Runs inference on videos."
    },
]

app = FastAPI(version='1.0', title='VIP-VIRA',
              description="API for processing videos",
              openapi_tags=tags_metadata)

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

app.include_router(
    health_check_controller.router,
    prefix="/Health",
    tags=["Health"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    detection_controller.router,
    prefix="/Inference",
    tags=["Inference"],
    responses={404: {"description": "Not found"}},
)
