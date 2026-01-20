from pyexpat import model
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import tv_tensors
from tokam2d_utils import TokamDataset
import random

class MosaicWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, output_size=(512, 512)):
        self.dataset = dataset
        self.output_size = output_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        indices = [idx] + [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        
        images = []
        boxes_list = []
        
        for k in indices:
            img, target = self.dataset[k]
            images.append(img)
            boxes_list.append(target["boxes"])

        c, h, w = images[0].shape
        mosaic_img = torch.full((c, h * 2, w * 2), 0.0, dtype=images[0].dtype)
        offsets = [(0, 0), (0, w), (h, 0), (h, w)]
        new_boxes = []

        for img, boxes, (off_y, off_x) in zip(images, boxes_list, offsets):
            mosaic_img[:, off_y:off_y+h, off_x:off_x+w] = img
            shifted_boxes = boxes.clone()
            shifted_boxes[:, 0] += off_x
            shifted_boxes[:, 2] += off_x
            shifted_boxes[:, 1] += off_y
            shifted_boxes[:, 3] += off_y
            new_boxes.append(shifted_boxes)

        all_boxes = torch.cat(new_boxes, dim=0)
        
        data = {
            "image": tv_tensors.Image(mosaic_img),
            "boxes": tv_tensors.BoundingBoxes(
                all_boxes,
                format="XYXY",
                canvas_size=(h * 2, w * 2)
            ),
            "labels": torch.ones((all_boxes.shape[0],), dtype=torch.int64)
        }
        
        mosaic_transforms = v2.Compose([
            v2.Resize(self.output_size),
            v2.SanitizeBoundingBoxes()
        ])
        
        transformed_data = mosaic_transforms(data)
        
        return transformed_data["image"], {
            "boxes": transformed_data["boxes"],
            "labels": transformed_data["labels"]
        }
    
def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return tuple(zip(*batch))

def train_model(training_dir):

    def get_transforms(train=True):
        transforms = []
        if train:
            transforms.append(v2.RandomHorizontalFlip(0.5))
            transforms.append(v2.RandomVerticalFlip(0.5))
            transforms.append(v2.RandomAffine(
                degrees=180, 
                translate=(0.2, 0.2), 
                scale=(0.3, 2.0),
                shear=10
            ))
        
        transforms.append(v2.ToDtype(torch.float32, scale=True))
        transforms.append(v2.SanitizeBoundingBoxes()) 
        
        return v2.Compose(transforms)

    final_transform = get_transforms()
    train_dataset = TokamDataset(training_dir)
    mosaic_dataset = MosaicWrapper(train_dataset, output_size=(512, 512))
    train_dataloader = torch.utils.data.DataLoader(
        mosaic_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True
    )

    if torch.cuda.is_available():
        print("Using GPU")

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, trainable_backbone_layers=1, box_score_thresh=0.01)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters())

    max_epochs = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in range(max_epochs):
        print(f"Epoch {i+1}/{max_epochs}")
        running_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            
            aug_images = []
            aug_targets = []

            for img, target in zip(images, targets):

                img = img.to(device)
                target = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}
                img_tv = tv_tensors.Image(img)
                target["boxes"] = tv_tensors.BoundingBoxes(
                    target["boxes"], 
                    format="XYXY", 
                    canvas_size=img.shape[-2:]
                )
                
                new_img, new_target = final_transform(img_tv, target)
                
                aug_images.append(new_img)
                aug_targets.append(new_target)
                
            optimizer.zero_grad()
            
            loss_dict = model(aug_images, aug_targets)
            
            full_loss = sum(loss for loss in loss_dict.values())
            running_loss += full_loss.item()
            
            full_loss.backward()
            optimizer.step()

    model.eval().to("cpu")
    return model
