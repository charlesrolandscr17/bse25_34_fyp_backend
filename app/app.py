from typing import Union
from fastapi import FastAPI

from parser.parser import word

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": word}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
