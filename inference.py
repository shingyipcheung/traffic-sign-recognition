import cv2
import torch
import pickle
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights
import torchvision.transforms as transforms


def draw(image, pred, label_map, threshold=0.6):
    img = image.mul(255).permute(1, 2, 0).byte().cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    boxes = pred['boxes'].cpu()
    labels = pred['labels'].cpu()
    scores = pred['scores'].cpu()

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            xmin, ymin, xmax, ymax = [int(coord) for coord in box]  # Convert to integers
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            label_text = f'{label_map[label.item()]} ({score:.2f})'
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (xmin, ymin - text_height - 4), (xmin + text_width, ymin), (0, 0, 255), -1)
            cv2.putText(img, label_text, (xmin, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_save_path = "trained_model.pth"
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

    with open('label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    num_classes = len(label_map)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    model.load_state_dict(torch.load(model_save_path))

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Detected signs", cv2.WINDOW_NORMAL)

    # img_transform = get_transform(train=False)
    img_transform = torchvision.transforms.Compose([
        transforms.ToTensor()
    ])
    model.eval()
    while True:
        ret, frame = cap.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_img = img_transform(img)
        input_tensor = input_img[None, ...]
        # Detect the lanes
        with torch.no_grad():
            prediction = model(images=input_tensor.to(device))

        processed_img = draw(input_tensor[0], prediction[0], label_map)

        cv2.imshow("Detected signs", processed_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
