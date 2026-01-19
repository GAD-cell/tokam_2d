# Seed:

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from tokam2d_utils import TokamDataset


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return tuple(zip(*batch))


def train_model(training_dir):
    # If you use `include_unlabeled=True`, you will have the frames from `turb_i`,
    # with no annotations included in the dataloader. This can be used for domain
    # adaptation, for instance using the skada library.
    train_dataset = TokamDataset(training_dir, include_unlabeled=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True
    )

    if torch.cuda.is_available():
        print("Using GPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = fasterrcnn_resnet50_fpn()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters())

    max_epochs = 1

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

    # Note that the model will be evaluated as is, and will be provided with
    # batch on CPU.
    model.eval().to("cpu")
    return model

```
