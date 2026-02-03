
YOLOv8-nano Model

Downloaded from: Ultralytics official repository
Model: yolov8n.pt
Size: ~6MB

For face detection, you can:
1. Use this general object detection model
2. Fine-tune on WIDER FACE dataset for better face detection
3. Download a pre-trained face detection model

For mask/helmet detection:
- Fine-tune on custom dataset with mask/helmet annotations
- Or use a pre-trained model from Roboflow/Ultralytics Hub

Model conversion to TFLite:
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='tflite')
