
# Diagnostic AI Models
## Introduction
This project has all of the ai models used in production, we have the following structure

maanz_ai_medical_models
<br>
|--- predictive
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- pathology
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- Mitotic Model
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- radiology
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- XrayPatchClassifier
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- XrayPatchDetector
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- XraySegmentor
<br>

#### Mitotic Model
Gives a probability from 0 to 1 for a square patch if it is Mitotic or Maybe Mitotic

#### Xray Patch Classifier
Classifies the disease associated in a rectangular patch

#### Xray Patch Detector
Detects rectangular patches in an xray patches which might be associated with a certain disease

#### Xray Segmentation
Pixel wise segmentation of lungs and abdomen in an xray image

## Build Instructions
1) pip install setuptools wheel
2) python setup.py bdist_wheel
