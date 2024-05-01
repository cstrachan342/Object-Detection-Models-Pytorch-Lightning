# Object-Detection-Models-Pytorch-Lightning
A collection of Object Detection models implemented using PyTorch Lightning, offering a streamlined approach to developing and implementing Object Detection algorithms.

## Installation

To use this repository please use the following code in terminal:

``` Python3
git clone https://github.com/cstrachan342/Object-Detection-Models-Pytorch-Lightning
```
```Python3
cd /Object-Detection-Models-Pytorch-Lightning
```
```Python3
pip install -q -r requirements.txt
```

## Usage

Once requirements are installed you can import Models ready for fitting within a Pytorch Lightning ```Trainer()```. This requires a DataModule already initialized for your Training and Validation Dataloaders. An example of this pipeline can be found in examples directory.

```Python3
from models.faster_rcnn_pytorch_lightning import CustomFasterRCNN
```

If you need to adapt the models for your specific tasks you can subclass and overwrite methods. For example if you want to change Loss of Faster-RCNN you can subclass and overwrite the common_step method as shown below:

```Python3
class NewCustomFasterRCNN(CustomFasterRCNN):
    def __init__(self):
        super().__init__()

    def common_step(self, batch, batch_idx):
        images = [x.to(self.device) for x in batch[0]]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch[1]]

        loss_dict = self.model(images, targets)
        loss_dict = {key: loss_dict[key] for key in ['loss_classifier', 'loss_objectness']}
        loss = sum(loss for loss in loss_dict.values())

        return loss, loss_dict
```

Alternatively you can enter the files and copy the class directly into your notebook and edit the code where you find necessary.

