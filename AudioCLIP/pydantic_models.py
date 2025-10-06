from pydantic import BaseModel, Field
from typing import List

class DatasetCreationResponse(BaseModel):
    classes: List[str] = Field(..., description="A list of unique class names.")
    video_queries: List[str] = Field(..., description="A list of video queries associated with the classes.")
