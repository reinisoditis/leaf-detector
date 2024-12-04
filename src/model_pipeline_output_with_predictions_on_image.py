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

# Colors for different model task predictions
COLORS = {
    'detection': (255, 0, 0),  # Blue for object detection
    'segmentation': (0, 255, 0),  # Green for segmentation
    'classification': (0, 0, 255)  # Red for classification
}


def predict_image(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs).squeeze().numpy()

    percentages = probabilities * 100
    formatted_percentages = ["{:.2f}%".format(p) for p in percentages]

    return formatted_percentages


def process_percentages(percentages):
    # Store results based on percentage_threshold, results > percentage_threshold will be taken
    percentage_threshold = 50
    conditions = []
    if float(percentages[0].strip('%')) > percentage_threshold:
        conditions.append('healthy')
    if float(percentages[1].strip('%')) > percentage_threshold:
        conditions.append('mineral')
    if float(percentages[2].strip('%')) > percentage_threshold:
        conditions.append('disease')
    return conditions


def add_label(image, text, position, color):
    """Helper function to add labels with background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    padding = 5
    cv2.rectangle(image,
                  (position[0] - padding, position[1] - text_size[1] - padding),
                  (position[0] + text_size[0] + padding, position[1] + padding),
                  (255, 255, 255),
                  -1)

    cv2.putText(image, text, position, font, font_scale, color, thickness)


def process_image(image_path):
    """Process a single image and return annotated image + results"""
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load the image at path: {image_path}")

    annotated_image = original_image.copy()
    results_data = []
    instance_counter = 1

    segmentation_overlay = np.zeros_like(original_image)

    # Object Detection. If needed override predict values (max_det, conf, iou..)
    detection_results = detection_model.predict(source=image_path, show=False, save=False, max_det=1000, conf=0.5, verbose=False)

    for det_result in detection_results:
        if det_result.boxes is not None:
            boxes = det_result.boxes.xyxy.cpu().numpy()
            detection_conf = det_result.boxes.conf.cpu().numpy()
            detection_cls = det_result.boxes.cls.cpu().numpy()

            for box_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                conf = detection_conf[box_idx]
                cls_id = detection_cls[box_idx]

                # Draw object detection box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), COLORS['detection'], 3)
                det_label = f"Det {cls_id}: {conf:.2f}"
                add_label(annotated_image, det_label, (x1, y1 - 10), COLORS['detection'])

                detected_region = original_image[y1:y2, x1:x2]

                # Segmentation on object detection model predicted coordinates
                # If needed override predict values (max_det, conf, iou..)
                seg_results = segmentation_model.predict(source=detected_region, show=False, save=False, verbose=False)

                for seg_result in seg_results:
                    if seg_result.masks is not None:
                        masks = seg_result.masks.data
                        seg_conf = seg_result.boxes.conf.cpu().numpy()
                        seg_cls = seg_result.boxes.cls.cpu().numpy()

                        for mask_idx, mask in enumerate(masks):
                            mask = mask.cpu().numpy()
                            mask = (mask * 255).astype(np.uint8)
                            mask = cv2.resize(mask, (detected_region.shape[1], detected_region.shape[0]))

                            mask_color = np.zeros_like(detected_region)
                            mask_color[:, :, 1] = mask

                            roi = segmentation_overlay[y1:y2, x1:x2]
                            cv2.addWeighted(roi, 1, mask_color, 0.5, 0, roi)
                            segmentation_overlay[y1:y2, x1:x2] = roi

                            local_x, local_y, w, h = cv2.boundingRect(mask)
                            global_x = x1 + local_x
                            global_y = y1 + local_y

                            cv2.rectangle(annotated_image,
                                          (global_x, global_y),
                                          (global_x + w, global_y + h),
                                          COLORS['classification'], 2)

                            seg_label = f"Seg {seg_cls[mask_idx]}: {seg_conf[mask_idx]:.2f}"
                            add_label(annotated_image, seg_label,
                                      (global_x, global_y + h + 20),
                                      COLORS['segmentation'])

                            cls_label = f"Cls {instance_counter}"
                            add_label(annotated_image, cls_label,
                                      (global_x, global_y - 10),
                                      COLORS['classification'])

                            instance = cv2.bitwise_and(detected_region, detected_region, mask=mask)
                            cropped_instance = instance[local_y:local_y + h, local_x:local_x + w]

                            # Classification [Class_1 = healthy,Class_2 = missing minerals ,Class_3 = disease]
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

    return annotated_image, {
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
            annotated_image, results = process_image(str(img_path))

            # Save annotated image
            output_image_path = Path(output_dir) / f"annotated_{img_path.name}"
            cv2.imwrite(str(output_image_path), annotated_image)

            # Store results
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