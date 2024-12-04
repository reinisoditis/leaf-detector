# Blackcurrant Leaf Disease Detection

-----

## Models:
    
    - Object detection:
        - yolov8x-detect.pt --imgsz=640
        - yolov8n-detect-1280.pt --imgsz=1280 (smaller model with bigger img input size, faster than 8x)

    - Segmentation:
        - yolov8n-seg.pt
        - yolov9c-seg.pt

    - Classification:
        - resnet50-model.pth

-----
## Model pipeline options:
1. Model pipeline with object detection, segmentation and classification models **with provided predictions on images**
   
   - python file name *model_pipeline_output_with_predictions_on_image.py*
   - Usage:
   
     - from root folder navigate to "src" folder `cd .\src\` 
       - folder with images and output images can be passed with arguments when running script: `python .\model_pipeline_output_with_predictions_on_image.py --input_path /path/to/your/input/images --output_dir /path/to/your/output/directory`
       - if arguments for script are not passed `python .\model_pipeline_output_with_predictions_on_image.py`, default values will be used
       - for script help (to see info about arguments) use - `python .\model_pipeline_output_with_predictions_on_image.py --help`


2. Model pipeline with object detection, segmentation and classification models **with NO predictions on images**
   
   - python file name *pipeline-structure-no-predictions-on-images.py*
   - Usage:
   
     - from root folder navigate to "src" folder `cd .\src\` 
       - folder with images and output images can be passed with arguments when running script: `python .\pipeline-structure-no-predictions-on-images.py --input_path /path/to/your/input/images --output_dir /path/to/your/output/directory`
       - if arguments for script are not passed `python .\pipeline-structure-no-predictions-on-images.py`, default values will be used
       - for script help (to see info about arguments) use - `python .\pipeline-structure-no-predictions-on-images.py --help`

-----

### For Developers

#### YOLO models predictions:
    
    - at the moment YOLO model prediction arguments can be changed only in code (max detections, confidence treshold, iou ..)
    - model pipeline results will be saved in .json file, in provided output_dir
    - .json file structure: 
    {
        "image_name.JPG": {
            "conditions": [
                "healthy",
                "mineral",
                "disease"
            ],
            "counts": {
            "healthy": 108,
            "mineral": 4,
            "disease": 6
            }
        }
    }
