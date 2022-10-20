<h3 align="center">
  <img src="img/unstructured_logo.png" height="200">
</h3>

<h3 align="center">
  <p>Open-Source Pre-Processing Tools for Unstructured Data</p>
</h3>

The `ml-inference` repo contains hosted model inference code for layout parsing models. These
models are invoked via API as part of the partitioning bricks in the `unstructured` package.

## Installation

To install the dependencies, run `make install`.
Run `make help` for a full list of install options.

## Getting Started

To get started with the layout parsing model, use the following commands:

```python
from ml_inference.inference.layout import DocumentLayout

layout = DocumentLayout.from_file("<filename>")
```

Once the model has detected the layout and OCR'd the document, you can run
`layout.pages[0].elements` to see the elements that were extracted from the first
page. You can convert a given element to a `dict` by running the `.to_dict()` method.

To use layout parsing API locally, run `make run-app-dev`.
Then you can hit the endpoint and upload a PDF file to see its layout with the command:
```
curl -X 'POST' 'http://127.0.0.1:8000/layout/pdf' -F 'file=@<your_pdf_file>' | jq -C . | less -R
```
You may need to install `poppler` for the endpoint with the command `brew install poppler`.

## Security Policy

See our [security policy](https://github.com/Unstructured-IO/ml-inference/security/policy) for
information on how to report security vulnerabilities.

## Learn more

| Section | Description |
|-|-|
| [Company Website](https://unstructured.io) | Unstructured.io product and company info |
