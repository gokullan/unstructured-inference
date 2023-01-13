import os

import pytest
from fastapi.testclient import TestClient

import unstructured_inference.models.detectron2 as detectron2
from unstructured_inference import models
from unstructured_inference.api import app
from unstructured_inference.inference.layout import DocumentLayout


@pytest.fixture
def sample_pdf_content():
    return """
    this is the content of a sample pdf file.
    Title: ...
    Author: ...
    """


class MockModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_layout_parsing_pdf_api(sample_pdf_content, tmpdir, monkeypatch):
    monkeypatch.setattr(models, "load_model", lambda *args, **kwargs: MockModel(*args, **kwargs))
    monkeypatch.setattr(detectron2, "is_detectron2_available", lambda *args: True)
    monkeypatch.setattr(
        DocumentLayout, "from_file", lambda *args, **kwargs: DocumentLayout.from_pages([])
    )

    filename = os.path.join(tmpdir.dirname, "sample.pdf")
    with open(filename, "w") as f:
        f.write(sample_pdf_content)

    client = TestClient(app)
    response = client.post("/layout/pdf", files={"file": (filename, open(filename, "rb"))})
    assert response.status_code == 200

    response = client.post(
        "/layout/pdf", files={"file": (filename, open(filename, "rb"))}, data={"model": "checkbox"}
    )
    assert response.status_code == 200

    response = client.post(
        "/layout/pdf",
        files={"file": (filename, open(filename, "rb"))},
        data={"model": "fake_model"},
    )
    assert response.status_code == 422

def test_layout_v02_api_parsing_image():

    filename = os.path.join("sample-docs", "test-image.jpg")
    
    client = TestClient(app)
    response = client.post("/layout/v0.2/image", 
                            headers={"Accept":"multipart/mixed"},
                            files=[
                                ("files", (filename, open(filename, "rb"), "image/png"))
                            ],
                            )
    # The example sent to the test contains 13 detections
    assert len(response.json()['Detections'])==13 
    # Each detection should have (x1,y1,x2,y2,probability,class) format
    assert len(response.json()['Detections'][0])==6
    assert response.status_code == 200

def test_layout_v02_local_parsing_image():
    filename = os.path.join("sample-docs", "test-image.jpg")
    from unstructured_inference.layout_model import local_inference

    detections = local_inference(filename,type='image')
    # The example sent to the test contains 13 detections
    assert len(detections['Detections'])==13
    # Each detection should have (x1,y1,x2,y2,probability,class) format
    assert len(detections['Detections'][0])==6

def test_layout_v02_local_parsing_pdf():
    filename = os.path.join("sample-docs", "loremipsum.pdf")
    from unstructured_inference.layout_model import local_inference

    detections = local_inference(filename,type='pdf')
    assert len(detections)==1
    # The example sent to the test contains 1 page
    assert len(detections['Detections'])==1
    assert len(detections['Detections'][0])==5
    # Each detection should have (x1,y1,x2,y2,probability,class) format
    assert len(detections['Detections'][0][0])==6

def test_healthcheck(monkeypatch):
    client = TestClient(app)
    response = client.get("/healthcheck")
    assert response.status_code == 200
