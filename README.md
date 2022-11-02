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

To build the Docker container, run `make docker-build`. Note that the CPU on an Apple M1 chip is limited to building `Detectron2` on Docker, you should build it on Linux. To run the API locally, use `make start-app-local`. You can stop the API with `make stop-app-local`. The API will run at `http:/127.0.0.1:5000`. You can view the swagger documentation at `http://127.0.0.1:5000/docs`.
Then you can hit the API endpoint and upload a PDF file to see its layout with the command:
```
curl -X 'POST' 'http://127.0.0.1:5000/layout/pdf' -F 'file=@<your_pdf_file>' | jq -C . | less -R
```

You can also choose the types of elements you want to return from the output of PDF parsing by passing a list of types to the `include_elems` parameter. For example, if you only want to return `Text` elements and `Title` elements, you can curl:
```
curl -X 'POST' 'http://127.0.0.1:5000/layout/pdf' \
-F 'file=@<your_pdf_file>' \
-F include_elems=Text \
-F include_elems=Title \
 | jq -C | less -R
```

If you are using an Apple M1 chip, use `make run-app-dev` instead of `make start-app-local` to start the API with hot reloading. The API will run at `http:/127.0.0.1:8000`.

## Security Policy

See our [security policy](https://github.com/Unstructured-IO/ml-inference/security/policy) for
information on how to report security vulnerabilities.

## Learn more

| Section | Description |
|-|-|
| [Company Website](https://unstructured.io) | Unstructured.io product and company info |
