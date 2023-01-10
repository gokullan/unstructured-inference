from fastapi import FastAPI, File, status, Request, UploadFile, Form, HTTPException
from unstructured_inference.inference.layout import DocumentLayout
from unstructured_inference.models import get_model
from typing import List
import tempfile
import cv2
import onnxruntime
from yolox_functions import preproc as preprocess
from yolox_functions import demo_postprocess,multiclass_nms
from visualize import vis
import numpy as np
import os

LAYOUT_CLASSES=["Caption","Footnote","Formula","List-item","Page-footer","Page-header","Picture","Section-header","Table","Text","Title"]
YOLOX_MODEL="path/..."
output_dir="..."

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

@app.post("/layout_v0.2/image")
async def layout_v02_parsing_image(
    file: UploadFile = File(),
    include_elems: List[str] = Form(default=ALL_ELEMS),
    model: str = Form(default=None),
):
    # The model was trained and exported with this shape
    # TODO: check other shapes for inference
    input_shape = (1024,768) 
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file.write(file.file.read())
        origin_img = cv2.imread(tmp_file)
        img,ratio = preprocess(origin_img,input_shape)

        session= onnxruntime.InferenceSession(YOLOX_MODEL)

        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], input_shape, p6=False)[0] #TODO: check for p6
 
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                            conf=0.3, class_names=LAYOUT_CLASSES)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, os.path.basename(tmp_file))
        cv2.imwrite(output_path, origin_img)
        #Returning dummy value temporarily
    return {"RESULT": "IMAGE_PROCESSED"}


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
async def healthcheck(request: Request):
    return {"healthcheck": "HEALTHCHECK STATUS: EVERYTHING OK!"}
