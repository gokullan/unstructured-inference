from fastapi import FastAPI, File, UploadFile
from ml_inference.inference.layout import DocumentLayout
import os
import shutil

app = FastAPI()


@app.post("/layout/pdf")
async def layout_parsing_pdf(file: UploadFile = File()):
    path = "uploaded_files/"
    os.mkdir(path)
    file_location = os.path.join(path, file.filename)
    with open(file_location, "wb") as f:
        f.write(file.file.read())

    layout = DocumentLayout.from_file(file_location)
    shutil.rmtree(path)
    pages_layout = [
        {"number": page.number, "elements": [element.to_dict() for element in page.elements]}
        for page in layout.pages
    ]

    return {"pages": pages_layout}
