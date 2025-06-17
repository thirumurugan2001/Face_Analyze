from pydantic import BaseModel

class UserFace(BaseModel):
    imageBase64: str