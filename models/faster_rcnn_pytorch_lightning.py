import pytorch_lightning as pl
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CustomFasterRCNN(pl.LightningModule):
    def __init__(self, lr, weight_decay, num_classes, batch_size, score_threshold, nms_iou_threshold):
        super().__init__()
        """
        FasterR-CNN Model using pre-trained resnet50 as backbone.
        
        lr - Learning rate for optimizer. Optimizer is set to Adam.
        weight_decay - Weight Decay for optimizer.
        num_classes - The number of classes model is detecting. Must add one to your amount of classes to represent the
                      background class. For example if you are trying to detect solely dogs in photos, the num_classes
                      will equal 2 (Dogs + Background).
        batch_size - Batch size of dataloader. This is needed to efficiently log the models loss scores.
        score_threshold - Threshold to use for output 'scores' when predicting images. Any predictions below the 
                          set threshold will be discarded. 
        nms_iou_threshold - The threshold score to use when completing NMS on model predictions.
        """

        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.batch_size = batch_size

        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', weights='DEFAULT', trainable_layers=5)
        anchor_generator = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (256,)),
                                           aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                        output_size=7,
                                                        sampling_ratio=2)

        self.model = FasterRCNN(backbone,
                                num_classes=self.num_classes,
                                rpn_anchor_generator=anchor_generator,
                                box_roi_pool=roi_pooler)

    def forward(self, images, targets):
        return self.model(images=images, targets=targets)

    def common_step(self, batch, batch_idx):
        images = [x.to(self.device) for x in batch[0]]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch[1]]

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        self.model.train()
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss.item())
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        self.model.train()
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss.item())
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        return [optimizer]

    def save_checkpoint(self):
        if os.path.exists('model_training') is False:
            os.mkdir('model_training')

        torch.save({'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict()
                    }, f'/content/model_training/{self.current_epoch}.pth')

    def on_train_epoch_end(self):

        train_loss_classifier = self.trainer.callback_metrics.get('train_loss_classifier', 'N/A')
        train_loss_objectness = self.trainer.callback_metrics.get('train_loss_objectness', 'N/A')
        train_loss_box_reg = self.trainer.callback_metrics.get('train_loss_box_reg', 'N/A')
        train_loss_rpn_box_reg = self.trainer.callback_metrics.get('train_loss_rpn_box_reg', 'N/A')

        validation_loss_classifier = self.trainer.callback_metrics.get('validation_loss_classifier', 'N/A')
        validation_loss_objectness = self.trainer.callback_metrics.get('validation_loss_objectness', 'N/A')
        validation_loss_box_reg = self.trainer.callback_metrics.get('validation_loss_box_reg', 'N/A')
        validation_rpn_box_reg = self.trainer.callback_metrics.get('validation_rpn_box_reg', 'N/A')

        self.save_checkpoint()

        print(f'Epoch {self.current_epoch}: '
              f'Train Loss Classifier: {train_loss_classifier}, '
              f'Train Loss Objectness: {train_loss_objectness}, '
              f'Train Loss Box Reg: {train_loss_box_reg}, '
              f'Train Loss Rpn Box Reg: {train_loss_rpn_box_reg}, '
              f'Validation Loss Classifier: {validation_loss_classifier}, '
              f'Validation Loss Objectness: {validation_loss_objectness} ',
              f'Validation Loss Box Reg: {validation_loss_box_reg}, '
              f'Validation Loss Rpn Box Reg: {validation_rpn_box_reg},')

    def decode_prediction(self, preds):
        boxes = preds['boxes']
        scores = preds['scores']
        labels = preds['labels']

        if self.score_threshold is not None:
            want = scores > self.score_threshold
            preds['boxes'] = boxes[want]
            preds['scores'] = scores[want]
            preds['labels'] = labels[want]

        if self.score_threshold is not None:
            want = scores > self.score_threshold
            boxes = boxes[want]
            scores = scores[want]
            labels = labels[want]

            keep = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=self.nms_iou_threshold)

            preds['boxes'] = preds['boxes'][keep]
            preds['scores'] = preds['scores'][keep]
            preds['labels'] = preds['labels'][keep]

        return preds

    def predict(self, dm, rand=True, images=None):
        self.model.eval().to(self.device)

        if rand:
            images = []
            for r in random.sample(range(0, len(dm.test_data)), 3):
                images.append(dm.test_data[r])

        fig, ax = plt.subplots(len(images), figsize=(10, 5 * len(images)))

        for v, (i, l) in enumerate(images):
            with torch.no_grad():
                outputs = self.model([i.to(self.device)])
                results = self.decode_prediction(*outputs)

            boxes = results['boxes'].cpu()
            labels = results['labels'].cpu()

            ax_curr = ax[v] if len(images) > 1 else ax
            ax_curr.imshow(i.permute(1, 2, 0))

            for box, label in zip(boxes, labels):
                x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
                ax_curr.add_patch(rect)
                if label:
                    ax_curr.text(x_min, y_min, label, color='r', fontsize=10, ha='left', va='bottom')
        return plt.show()
