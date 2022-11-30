from fastapi import APIRouter, Response

router = APIRouter()


@router.get('/')
async def check_health():
    return Response(status_code=200)


@router.post("/")
async def check_health():
    return {"message": "hello world"}
