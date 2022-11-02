from fastapi import FastAPI, File, status, Request, UploadFile, Form
from ml_inference.inference.layout import DocumentLayout
from typing import List
import tempfile

app = FastAPI()

ALL_ELEMS = "_ALL"


@app.post("/layout/pdf")
async def layout_parsing_pdf(
    file: UploadFile = File(), include_elems: List[str] = Form(default=ALL_ELEMS)
):
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file.write(file.file.read())
        layout = DocumentLayout.from_file(tmp_file.name)
    pages_layout = [
        {
            "number": page.number,
            "elements": [
                element.to_dict()
                for element in page.elements
                if element.type in include_elems or include_elems == ALL_ELEMS
            ],
        }
        for page in layout.pages
    ]

    return {"pages": pages_layout}


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
async def healthcheck(request: Request):
    return {"healthcheck": "HEALTHCHECK STATUS: EVERYTHING OK!"}
