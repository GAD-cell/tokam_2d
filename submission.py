import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import v2

from tokam2d_utils import TokamDataset


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return tuple(zip(*batch))

def get_transforms(train=True):
    transforms = []
    if train:
        transforms.append(v2.RandomHorizontalFlip(0.5))
        transforms.append(v2.RandomVerticalFlip(0.5))
        transforms.append(v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)))
    
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    transforms.append(v2.SanitizeBoundingBoxes()) 
    
    return v2.Compose(transforms)

final_transform = get_transforms()

def train_model(training_dir):
    train_dataset = TokamDataset(training_dir)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True
    )

    if torch.cuda.is_available():
        print("Using GPU")

    model = fasterrcnn_resnet50_fpn()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters())

    max_epochs = 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in range(max_epochs):
        print(f"Epoch {i+1}/{max_epochs}")
        for images, targets in train_dataloader:
            images = [im.to(device) for im in images]
            targets = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()} for t in targets
            ]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            full_loss = sum(loss for loss in loss_dict.values())
            full_loss.backward()
            optimizer.step()

    model.eval().to("cpu")
    return model
