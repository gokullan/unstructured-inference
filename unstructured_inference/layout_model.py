import tempfile
import os
import wget # TODO: to migrate to huggingface
import cv2
import onnxruntime
from unstructured_inference.yolox_functions import preproc as preprocess
from unstructured_inference.yolox_functions import demo_postprocess,multiclass_nms
from unstructured_inference.visualize import vis
import numpy as np
import os
import wget
from PIL import Image
from pdf2image import convert_from_bytes,convert_from_path
from unstructured_inference.inference.layout import LayoutElement, PageLayout, DocumentLayout

S3_SOURCE="https://utic-dev-tech-fixtures.s3.us-east-2.amazonaws.com/layout_model/yolox_l0.05.onnx"
LAYOUT_CLASSES=["Caption","Footnote","Formula","List-item","Page-footer","Page-header","Picture","Section-header","Table","Text","Title"]
YOLOX_MODEL="/Users/benjamin/Documents/unstructured-inference/.models/yolox_l0.05.onnx"
output_dir="outputs/"


def local_inference(filename,type='image'):
    if not os.path.exists(".models/yolox_l0.05.onnx"):
        wget.download(S3_SOURCE,'.models/yolox_l0.05.onnx')

    pages_paths = []
    detections = []
    detectedDocument = None
    if type=='pdf':
        with tempfile.TemporaryDirectory() as tmp_folder:
            pages_paths = convert_from_path(filename, dpi= 500,
                                            output_folder=tmp_folder,
                                            paths_only=True)
            for i,path in enumerate(pages_paths):
                # Return a dict of {n-->PageLayoutDocument}
                detections.append( image_processing(path,page_number=i) )
            detectedDocument = DocumentLayout (detections)
    else:
        # Return a PageLayoutDocument
        detections = image_processing(filename)
        detectedDocument = DocumentLayout ( [detections])
      
    return detectedDocument

def image_processing(page,page_number=0):

    # The model was trained and exported with this shape
    # TODO: check other shapes for inference
    input_shape = (1024,768) 
    origin_img = cv2.imread(page)
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

    elements=[]
    for det in dets:
        detection = det.tolist()
        detection[-1] = LAYOUT_CLASSES[int(detection[-1])]
        element = LayoutElement(type=detection[-1],
                                coordinates=[ (detection[0],detection[1]),
                                              (detection[2],detection[3]) ],
                                text=" ") # TODO: get text from document
        
        elements.append(element)

    page = PageLayout(number=page_number,image=origin_img,layout=elements)

    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    #output_path = os.path.join(output_dir, os.path.basename(tmp_file.name),".jpg") #the  tmp_file laks of extension
    #cv2.imwrite(output_path, origin_img)

    return page
