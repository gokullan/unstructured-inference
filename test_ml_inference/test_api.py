import pytest
import os

from fastapi.testclient import TestClient
from ml_inference.api import app

from ml_inference.inference.layout import DocumentLayout
import ml_inference.models.detectron2 as detectron2


@pytest.fixture
def sample_pdf_content():
    return """
    this is the content of a sample pdf file.
    Title: ...
    Author: ...
    """


def test_layout_parsing_pdf_api(sample_pdf_content, tmpdir, monkeypatch):
    monkeypatch.setattr(detectron2, "is_detectron2_available", lambda *args: True)
    monkeypatch.setattr(DocumentLayout, "from_file", lambda *args: DocumentLayout.from_pages([]))

    filename = os.path.join(tmpdir.dirname, "sample.pdf")
    with open(filename, "w") as f:
        f.write(sample_pdf_content)

    client = TestClient(app)
    response = client.post("/layout/pdf", files={"file": (filename, open(filename, "rb"))})
    assert response.status_code == 200


def test_healthcheck(monkeypatch):
    client = TestClient(app)
    response = client.get("/healthcheck")
    assert response.status_code == 200
