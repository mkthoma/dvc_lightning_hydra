import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, MeanMetric, MaxMetric
import timm

class TimmClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 10,
        pretrained: bool = True,
        lr: float = 1e-3,
        optimizer: str = "Adam",
        optimizer_params: dict = {},
        scheduler: str = "ReduceLROnPlateau",
        scheduler_params: dict = {},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

        # Load pre-trained model
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Multi-class accuracy
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.optimizer)
        optimizer = optimizer_class(self.parameters(), lr=self.lr, **self.optimizer_params)

        scheduler_class = getattr(torch.optim.lr_scheduler, self.scheduler)
        scheduler = scheduler_class(optimizer, **self.scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/acc"
        }

# Example usage
if __name__ == "__main__":
    model = TimmClassifier(
        model_name="resnet50",
        num_classes=10,
        pretrained=True,
        lr=1e-3,
        optimizer="Adam",
        optimizer_params={"weight_decay": 1e-5},
        scheduler="ReduceLROnPlateau",
        scheduler_params={"factor": 0.1, "patience": 5, "verbose": True},
    )
    print(model)