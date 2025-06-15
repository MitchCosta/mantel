from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import train_som
from schemas import SOMRequest, SOMResponse
from utils import save_image
import numpy as np
import uuid
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI(title="Kohonen SOM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/train", response_model=SOMResponse)
def train_som_api(request: SOMRequest):
    try:
        input_array = np.array(request.input_data)
        output = train_som(input_array, request.iterations, request.width, request.height)
        filename = f"som_image.png"            # filename = f"som_{uuid.uuid4().hex[:8]}.png"
        image_path = save_image(output, filename=filename)
        return SOMResponse(image_path=image_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/get_image")
async def get_image():
    image_path = Path("outputs/som_image.png")
    if not image_path.is_file():
        return {"error": "Image not found on the server"}
    return FileResponse(image_path)