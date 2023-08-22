import cv2
import numpy as np
import torch
import torchvision
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, random_split
from pycocotools.coco import COCO
import os
import pickle
from visualization import visualize_dataset, visualize_predictions

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights, \
    FasterRCNN_MobileNet_V3_Large_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights

import albumentations as A
from albumentations.pytorch import ToTensorV2


def collate_fn_coco(batch):
    return tuple(zip(*batch))


def get_transform(train=True):
    if train:
        return A.Compose([
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    else:
        return A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))


class TrafficSignDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.index_map = {i: key for i, key in enumerate(self.coco.imgToAnns.keys())}
        self.label_map = self.get_label_map()

    @staticmethod
    def _get_annotation(img_id, coco_annotation):
        num_objs = len(coco_annotation)
        boxes = []
        areas = []
        labels = []
        for ann in coco_annotation:
            boxes.append(ann['bbox'])
            areas.append(ann['area'])
            labels.append(ann['category_id'])

        return {
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id,
            'area': areas,
            'iscrowd': [0] * num_objs,
        }

    def __getitem__(self, index):
        img_id = self.index_map[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)

        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.root, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        annotation = self._get_annotation(img_id, coco_annotation)
        bboxes = annotation['boxes']
        labels = annotation['labels']
        areas = annotation['area']
        iscrowd = annotation['iscrowd']
        if self.transforms:
            # if you got an error
            # https://github.com/albumentations-team/albumentations/issues/459
            transformed = self.transforms(image=img, bboxes=bboxes, class_labels=labels)
            img = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['class_labels']

        for i, box in enumerate(bboxes):
            bboxes[i] = [box[0], box[1], box[2] + box[0], box[3] + box[1]]
        annotation = {
            'boxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.LongTensor(labels),
            'image_id': torch.tensor([img_id]),
            'area': torch.tensor(areas, dtype=torch.float32),
            'iscrowd': torch.zeros(iscrowd, dtype=torch.int64),
        }

        return img.div(255), annotation

    def get_label_map(self):
        label_map = {}
        for category in self.coco.cats.values():
            label_map[category['id']] = category['name']
        return label_map

    def __len__(self):
        return len(self.index_map)


def train(model, data_loader, device, optimizer):
    model.train()
    i = 0
    for imgs, annotations in data_loader:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f'Iteration: {i}/{len(data_loader)}, Loss: {losses}')


def train_test_split(dataset, train_ratio):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_indices, test_indices = random_split(range(len(dataset)), [train_size, test_size])

    train_dataset = torch.utils.data.Subset(dataset, train_indices.indices)
    train_dataset.coco = dataset.coco
    train_dataset.label_map = dataset.label_map
    test_dataset = torch.utils.data.Subset(dataset, test_indices.indices)
    test_dataset.coco = dataset.coco
    test_dataset.label_map = dataset.label_map

    return train_dataset, test_dataset


def evaluate(model, data_loader, device):
    detections = []
    coco_gt = data_loader.dataset.coco

    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                img_id = targets[i]['image_id'].item()
                for bbox, score, label in zip(output['boxes'], output['scores'], output['labels']):
                    xmin, ymin, xmax, ymax = bbox.tolist()
                    width_bbox = xmax - xmin
                    height_bbox = ymax - ymin
                    category_id = label.item()

                    detections.append({
                        'image_id': img_id,
                        'category_id': category_id,
                        'bbox': [xmin, ymin, width_bbox, height_bbox],
                        'score': score.item()
                    })

    coco_results = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_results, 'bbox')
    coco_eval.params.imgIds = list(np.unique([ann['image_id'] for ann in detections]))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = coco_eval.stats[0]
    print(f"Test mAP: {mAP:.4f}")


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Train = True
    # Load COCO dataset
    full_dataset = TrafficSignDataset(root='./', annotation='./result.json', transforms=get_transform(train=Train))
    coco = full_dataset.coco
    counts = {coco.cats[key]['name']: len(imgs) for key, imgs in coco.catToImgs.items()}
    print(len(full_dataset))
    print(counts)
    print(sum(v for v in counts.values()))
    # Split the dataset into train and validation sets
    train_dataset, test_dataset = train_test_split(full_dataset, 0.8)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0,
                              collate_fn=collate_fn_coco)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0,
                             collate_fn=collate_fn_coco)

    label_map = full_dataset.get_label_map()
    # visualize_dataset(train_dataset, label_map)

    model_save_path = "trained_model.pth"
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    num_classes = len(full_dataset.coco.getCatIds())
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    with open('label_map.pkl', 'wb') as f:
        pickle.dump(label_map, f)

    if Train:
        # parameters
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=0.001)

        num_epochs = 50
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch+1}/{num_epochs}")
            print("Training")
            train(model, train_loader, device, optimizer)
        torch.save(model.state_dict(), model_save_path)
    else:
        model.load_state_dict(torch.load(model_save_path))
        evaluate(model, test_loader, device)
        # Perform inference on the validation set
        # for images, targets in test_loader:
        #     test_images = [img.to(device) for img in images]
        #     model.eval()
        #     with torch.no_grad():
        #         test_predictions = model(test_images)
        #
        #     # Visualize predictions
        #     visualize_predictions(test_images, targets, test_predictions, label_map)


if __name__ == '__main__':
    main()
