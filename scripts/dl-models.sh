#!/bin/bash

set -euo pipefail

aws s3 cp s3://utic-dev-models/oer_checkbox/detectron2_oer_checkbox.json .models/detectron2_oer_checkbox.json
aws s3 cp s3://utic-dev-models/oer_checkbox/detectron2_finetuned_oer_checkbox.pth .models/detectron2_finetuned_oer_checkbox.pth
