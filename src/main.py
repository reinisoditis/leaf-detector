from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import pandas as pd


class MultiLabelModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelModel, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


classification_model = torch.load('./resnet50-model.pth')
classification_model.eval() 

transform = transforms.Compose([
    transforms.Resize((128, 128)),
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
    formatted_percentages = ["{:.2f}%".format(p) for p in percentages]

    return formatted_percentages

yolo_model = YOLO('./yolov8n-seg.pt')
image_path = './test_images/DJI_0383.JPG'


results = yolo_model.predict(source=image_path, show=False, save=False)

original_image = cv2.imread(image_path)

if original_image is None:
    raise ValueError(f"Could not load the image at path: {image_path}")


annotated_image = original_image.copy()
results_data = []

for idx, result in enumerate(results):
    if result.masks is not None:
        masks = result.masks.data

        for i, mask in enumerate(masks):
            mask = mask.cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))

            mask_color = np.zeros_like(original_image)
            mask_color[:, :, 1] = mask
            overlay = cv2.addWeighted(annotated_image, 1, mask_color, 0.3, 0)

            annotated_image = overlay
            instance = cv2.bitwise_and(original_image, original_image, mask=mask)
            x, y, w, h = cv2.boundingRect(mask)
            cropped_instance = instance[y:y+h, x:x+w]

            percentages = predict_image(cropped_instance, classification_model)

            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = str(i + 1)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x + 5
            text_y = y + text_size[1] + 5
            cv2.putText(annotated_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            results_data.append([i + 1] + percentages)


annotated_image_path = './results/annotated_leaves.png'
cv2.imwrite(annotated_image_path, annotated_image)
print(f'Saved annotated image to {annotated_image_path}')

csv_columns = ['Instance'] + [f'Class_{j+1}' for j in range(len(percentages))]
results_df = pd.DataFrame(results_data, columns=csv_columns)
csv_path = './results/leaf_classification_results.csv'
results_df.to_csv(csv_path, index=False)
print(f'Saved classification results to {csv_path}')
