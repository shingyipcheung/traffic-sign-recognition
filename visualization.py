import cv2


def visualize_dataset(dataset, label_map):
    for i in range(len(dataset)):
        image, target = dataset[i]
        img = image.mul(255).permute(1, 2, 0).byte().cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        boxes = target['boxes'].cpu()
        labels = target['labels'].cpu()

        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = [int(coord) for coord in box]  # Convert to integers
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            label_text = f'{label_map[label.item()]}'
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (xmin, ymin - text_height - 4), (xmin + text_width, ymin), (0, 0, 255), -1)
            cv2.putText(img, label_text, (xmin, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(f'Image {i}', img)
        cv2.waitKey()
        # output_path = f"./augmentation/{target['image_id'].item()}.png"
        # cv2.imwrite(output_path, img)
        cv2.destroyWindow(f'Image {i}')


def visualize_predictions(images, targets, predictions, label_map, threshold=0.40):
    for i, (image, target, pred) in enumerate(zip(images, targets, predictions)):
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

        output_path = f"./prediction/{target['image_id'].item()}.png"
        cv2.imwrite(output_path, img)
