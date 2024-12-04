from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import os
from pathlib import Path
import argparse
import time
import json


class MultiLabelModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelModel, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


# Load models
detection_model = YOLO('./yolov8n-detect-1280.pt')
segmentation_model = YOLO('./yolov9c-seg.pt')
# use cuda if available, if not then set device to cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classification_model = torch.load('./resnet50-model.pth', map_location=device)
classification_model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def predict_image(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs).squeeze().numpy()

    percentages = probabilities * 100
    formatted_percentages = ["{:.2f}".format(p) for p in percentages]

    return formatted_percentages


def process_percentages(percentages):
    # Store results based on percentage_threshold, results > percentage_threshold will be taken
    percentage_threshold = 50
    conditions = []

    if float(percentages[0]) >= percentage_threshold:
        conditions.append('healthy')
    if float(percentages[1]) >= percentage_threshold:
        conditions.append('mineral')
    if float(percentages[2]) >= percentage_threshold:
        conditions.append('disease')
    return conditions


def process_image(image_path):
    """Process a single image and return results"""
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load the image at path: {image_path}")

    results_data = []
    instance_counter = 1

    # Object Detection. If needed override predict values (max_det, conf, iou..)
    detection_results = detection_model.predict(source=image_path, show=False, save=False, max_det=1000, conf=0.5, verbose=False)

    for det_result in detection_results:
        if det_result.boxes is not None:
            boxes = det_result.boxes.xyxy.cpu().numpy()
            for box_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                detected_region = original_image[y1:y2, x1:x2]

                # Segmentation on object detection model predicted coordinates
                # If needed override predict values (max_det, conf, iou..)
                seg_results = segmentation_model.predict(source=detected_region, show=False, save=False, verbose=False)

                for seg_result in seg_results:
                    if seg_result.masks is not None:
                        masks = seg_result.masks.data
                        for mask in masks:
                            mask = mask.cpu().numpy()
                            mask = (mask * 255).astype(np.uint8)
                            mask = cv2.resize(mask, (detected_region.shape[1], detected_region.shape[0]))

                            local_x, local_y, w, h = cv2.boundingRect(mask)
                            instance = cv2.bitwise_and(detected_region, detected_region, mask=mask)
                            cropped_instance = instance[local_y:local_y + h, local_x:local_x + w]

                            # Classification
                            percentages = predict_image(cropped_instance, classification_model)

                            # Store results
                            results_data.append([instance_counter] + percentages)
                            instance_counter += 1

    conditions_count = {'healthy': 0, 'mineral': 0, 'disease': 0}
    all_conditions = set()

    for result in results_data:
        instance_conditions = process_percentages(result[1:])
        for condition in instance_conditions:
            conditions_count[condition] += 1
            all_conditions.add(condition)

    return {
        'conditions': list(all_conditions),
        'counts': {k: v for k, v in conditions_count.items() if v > 0}
    }


def save_results(results_dict, output_dir):
    """Save all results to a single JSON file"""
    # .json file name is set here
    json_path = Path(output_dir) / 'leaf_conditions.json'
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f'Saved all results to {json_path}')


def process_input(input_path, output_dir):
    """Process either a single image or a directory of images"""
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_path)

    # Dictionary to store all results
    all_results = {}

    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_extensions = ('.jpg', '.jpeg', '.png')
        image_paths = [f for f in input_path.glob('*') if f.suffix.lower() in image_extensions]

    for img_path in image_paths:
        try:
            print(f"Processing {img_path.name}...")

            # Process the image
            results = process_image(str(img_path))

            # Store results in the dictionary
            all_results[img_path.name] = results

        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            continue

    # Save all results to a single JSON file
    save_results(all_results, output_dir)


# Main execution
if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='Process image(s) through detection-segmentation-classification pipeline')
    parser.add_argument('--input_path', default='./folder_with_images_to_process', help='Path to input image or directory')
    parser.add_argument('--output_dir', default='./results_for', help='Path to output directory')

    args = parser.parse_args()
    process_input(args.input_path, args.output_dir)

    print("--- %s seconds ---" % (time.time() - start_time))