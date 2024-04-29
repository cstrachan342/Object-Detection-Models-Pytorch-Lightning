from transformers import DetrForObjectDetection
import pytorch_lightning as pl
import torch
import os


class CustomDETR(pl.LightningModule):
    def __init__(self, lr, lr_backbone, num_classes, weight_decay, batch_size):
        super().__init__()
        """
        DETR Model using pre-trained resnet50 as backbone.

        lr - Learning rate for the optimizer of params not in the backbone. Optimizer is set as AdamW.
.       lr_backbone - Learning rate for optimizer controlling weights in the backbone of the model.
        weight_decay - Weight Decay for optimizers
        num_classes - The number of classes model is detecting. Must add one to your amount of classes to represent the
                      background class. For example if you are trying to detect solely dogs in photos, the num_classes
                      will equal 2 (Dogs + Background).
        batch_size - Batch size of dataloader. This is needed to efficiently log the models loss scores.
        """

        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path='facebook/detr-resnet-50',
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            revision="no_timm"
        )

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.batch_size = batch_size

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):

        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def save_checkpoint(self):

        if os.path.exists('model_training') is False:
            os.mkdir('model_training')

        torch.save({'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict()
                    }, f'/content/model_training/{self.current_epoch}.pth')

    def on_train_epoch_end(self):

        train_loss_ce = self.trainer.callback_metrics.get('train_loss_ce', 'N/A')
        train_loss_bbox = self.trainer.callback_metrics.get('train_loss_bbox', 'N/A')
        train_loss_giou = self.trainer.callback_metrics.get('train_loss_giou', 'N/A')
        train_cardinality_error = self.trainer.callback_metrics.get('train_cardinality_error', 'N/A')

        validation_loss_ce = self.trainer.callback_metrics.get('validation_loss_ce', 'N/A')
        validation_loss_bbox = self.trainer.callback_metrics.get('validation_loss_bbox', 'N/A')
        validation_loss_giou = self.trainer.callback_metrics.get('validation_loss_giou', 'N/A')
        validation_cardinality_error = self.trainer.callback_metrics.get('validation_cardinality_error', 'N/A')

        self.save_checkpoint()

        print(f'Epoch {self.current_epoch}: '
              f'Train Loss CE: {train_loss_ce}, '
              f'Train Loss BBox: {train_loss_bbox}, '
              f'Train Loss Giou: {train_loss_giou}, '
              f'Train Cardinality Error: {train_cardinality_error}, '
              f'Validation Loss CE: {validation_loss_ce}, '
              f'Validation Loss BBox: {validation_loss_bbox}, '
              f'Validation Loss Giou: {validation_loss_giou}, '
              f'Validation Cardinality Error: {validation_cardinality_error}')
