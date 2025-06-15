from pydantic import BaseModel, Field
from typing import List

class SOMRequest(BaseModel):
    input_data: List[List[float]]
    width: int = Field(gt=0, example=10)
    height: int = Field(gt=0, example=10)
    iterations: int = Field(gt=0, example=100)

class SOMResponse(BaseModel):
    image_path: str
