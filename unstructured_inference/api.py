import tempfile
from typing import List, Union

from fastapi import (FastAPI, File, Form, HTTPException, Request, UploadFile,
                     status)

from unstructured_inference.inference.layout import DocumentLayout
from unstructured_inference.layout_model import local_inference
from unstructured_inference.models import get_model

S3_SOURCE = (
    "https://utic-dev-tech-fixtures.s3.us-east-2.amazonaws.com/layout_model/yolox_l0.05.onnx"
)
LAYOUT_CLASSES = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]
YOLOX_MODEL = "/Users/benjamin/Documents/unstructured-inference/.models/yolox_l0.05.onnx"
output_dir = "outputs/"

app = FastAPI()

ALL_ELEMS = "_ALL"


@app.post("/layout/pdf")
async def layout_parsing_pdf(
    file: UploadFile = File(),
    include_elems: List[str] = Form(default=ALL_ELEMS),
    model: str = Form(default=None),
):
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file.write(file.file.read())
        if model is None:
            layout = DocumentLayout.from_file(tmp_file.name)
        else:
            try:
                detector = get_model(model)
            except ValueError as e:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))
            layout = DocumentLayout.from_file(tmp_file.name, model=detector)
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


@app.post("/layout/v0.2/image")
async def layout_v02_parsing_image(
    request: Request,
    files: Union[List[UploadFile], None] = File(default=None),
):

    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file.write(files[0].file.read())
        detections = local_inference(tmp_file.name, type="image", to_json=True)

    return detections  # Already a dictionary


@app.post("/layout/v0.2/pdf")
async def layout_v02_parsing_pdf(
    request: Request,
    files: Union[List[UploadFile], None] = File(default=None),
):

    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file.write(files[0].file.read())
        detections = local_inference(tmp_file.name, type="pdf", to_json=True)

    return detections  # Already a dictionary


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
async def healthcheck(request: Request):
    return {"healthcheck": "HEALTHCHECK STATUS: EVERYTHING OK!"}
