# How to participate

You should submit a `submission.py` file which contains a `train_model` function, taking a `training_dir` parameter. This function should return an instance of a `pytorch.Model` class, which is trained with the given data.

This model will then be used with the following evaluation loop:

```python
def evaluate_model(model, test_dir):
    from tokam2d_utils import TokamDataset

    eval_dataset = TokamDataset(test_dir)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=2, collate_fn=collate_fn
    )

    model.eval()
    res = []
    for X, y in eval_dataloader:
        with torch.no_grad():
            # Here, y_pred should be a list of dict with keys:
            # - "boxes": Tensor of shape (num_boxes, 4)
            # - "scores": Tensor of shape (num_boxes,)
            y_pred = model(X)
        res.extend(y_pred)

    return res
```

See the "Seed" page for the outline of a `train_model` function, with the
expected features to leverage GPU training.

Note that you can use any library you want in your submission, as long as you
require it to the organizers in advance.
You also have access to the `TokamDataset` class, provided in `tokam2d_utils`.
This gives you a `torch.Dataset` class with one item per frame.
Using `include_unlabeled=True` will also give you access to frames with no
annotations from the `turb_i` simulation, to use for domain adapation.
