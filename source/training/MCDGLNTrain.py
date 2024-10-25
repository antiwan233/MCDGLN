from omegaconf import DictConfig
from typing import List
import logging
import torch
from torch_geometric.loader import DataLoader
from torch.optim import Optimizer
from source.components.lr_scheduler import LRScheduler
from source.utils import count_params, TotalMeter, EarlyStopping
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
import wandb
from tqdm import tqdm
import os


class MCDGLNTrain:

    def __init__(self,
                 cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[DataLoader],
                 logger: logging.Logger,
                 fold: int = None) -> None:

        # 在构造函数中初始化一些指标
        self.test_accuracy = None
        self.val_accuracy = None
        self.train_accuracy = None
        self.test_loss = None
        self.val_loss = None
        self.train_loss = None
        self.current_step = 0

        self.fold = fold

        self.device = torch.device('cuda:' + str(cfg.cuda) if torch.cuda.is_available() else 'cpu')

        self.cfg = cfg
        self.model = model
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.train_dataloader, self.val_dataloader = dataloaders
        self.logger = logger

        # 计算模型参数数量
        self.logger.info(f'#model params: {count_params(self.model)}')

        self.epochs = cfg.training.epochs
        self.total_steps = cfg.training.total_steps

        # 损失函数
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

        # 设置保存路径
        self.save_path = Path(cfg.log_path) / cfg.time_id

        self.init_meters()

    # 初始化训练过程的指标
    def init_meters(self):
        self.train_loss, self.val_loss, self.test_loss, \
        self.train_accuracy, self.val_accuracy, self.test_accuracy = [
            TotalMeter() for _ in range(6)]

    # 重置训练过程的指标
    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    # 单个epoch内的训练过程
    # 一个epoch分为数个step，每个step遍历一个batch
    def train_per_epoch(self, optimizer, lr_scheduler):

        self.model.train()

        for _, data in enumerate(self.train_dataloader):
            # 更新step, 用于lr_scheduler
            self.current_step += 1
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            x, windows, batch, labels = data.x.to(self.device), data.windows.to(self.device), \
                                           data.batch.to(self.device), data.y.to(self.device)

            output = self.model(x, windows, batch)
            predict = torch.argmax(output, dim=1)
            loss = self.loss_fn(output, labels.long())

            self.train_loss.update_with_weight(loss.item(), labels.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_accuracy = round(accuracy_score(labels.cpu().numpy(), predict.cpu().numpy()), 5)
            self.train_accuracy.update_with_weight(batch_accuracy, labels.shape[0])

            # if self.cfg.use_wandb:
            #     wandb.log({"LR": lr_scheduler.lr,
            #                "Iter loss": loss.item()})

    def val_per_epoch(self, val_dataloader):

        self.model.eval()

        with torch.no_grad():
            for _, data in enumerate(val_dataloader):
                x, windows, batch, labels = data.x.to(self.device), data.windows.to(self.device), \
                                            data.batch.to(self.device), data.y.to(self.device)

                output = self.model(x, windows, batch)
                predict = torch.argmax(output, dim=1)
                loss = self.loss_fn(output, labels.long())
                self.val_loss.update_with_weight(loss.item(), labels.shape[0])

                batch_accuracy = round(accuracy_score(labels.cpu().numpy(), predict.cpu().numpy()), 5)
                self.val_accuracy.update_with_weight(batch_accuracy, labels.shape[0])

    # 测试集
    def test(self, test_dataloader):

        val_labels = []
        val_preds = []

        self.model.eval()

        with torch.no_grad():
            for _, data in enumerate(test_dataloader):
                x, windows, batch, labels = data.x.to(self.device), data.windows.to(self.device), \
                                            data.batch.to(self.device), data.y.to(self.device)

                output = self.model(x, windows, batch)
                predict = torch.argmax(output, dim=1)
                loss = self.loss_fn(output, labels.long())
                self.test_loss.update_with_weight(loss.item(), labels.shape[0])

                batch_accuracy = round(accuracy_score(labels.cpu().numpy(), predict.cpu().numpy()), 5)
                self.test_accuracy.update_with_weight(batch_accuracy, labels.shape[0])
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predict.cpu().numpy())

        acc = round(accuracy_score(val_labels, val_preds), 5)
        auc = round(roc_auc_score(val_labels, val_preds), 5)
        f1 = round(f1_score(val_labels, val_preds), 5)
        precision = round(precision_score(val_labels, val_preds), 5)
        recall = round(recall_score(val_labels, val_preds), 5)
        tn, fp, fn, tp = confusion_matrix(val_labels, val_preds).ravel()

        specificity = round(tn / (tn + fp), 5)
        sensitivity = round(tp / (tp + fn), 5)

        results = {
            "Accuracy": acc,
            "AUC": auc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "Sensitivity": sensitivity
        }

        return results

    def train(self):

        # 早停
        if self.fold is not None:
            ckpt_path = os.path.join(self.cfg.ckpt_path, f"fold_{self.fold}_checkpoint.pth")
        else:
            ckpt_path = os.path.join(self.cfg.ckpt_path, "checkpoint.pth")

        early_stopping = EarlyStopping(patience=self.cfg.training.early_stop,
                                       verbose=True,
                                       path=ckpt_path)

        for epoch in range(1, self.epochs + 1):

            self.reset_meters()

            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])

            self.val_per_epoch(self.val_dataloader)

            self.logger.info(" | ".join([
                f"Epoch: {epoch}",
                f"Train Loss: {self.train_loss.avg: .3f}",
                f"Train Accuracy: {self.train_accuracy.avg: .3f}%",
                f"Val Loss: {self.val_loss.avg: .3f}",
                f"Val Accuracy: {self.val_accuracy.avg: .3f}%"
            ]))

            if self.cfg.use_wandb:
                wandb.log({
                    "train/Epoch": epoch,
                    "train/Train Loss": self.train_loss.avg,
                    "train/Train Accuracy": self.train_accuracy.avg,
                    "val/Val Loss": self.val_loss.avg,
                    "val/Val Accuracy": self.val_accuracy.avg
                })

            # 判断早停
            early_stopping(self.val_accuracy.avg, self.model)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

        # 测试集的指标
        self.model.load_state_dict(torch.load(ckpt_path))

        self.logger.info(f"Load model from {ckpt_path}, start testing...")

        results = self.test(self.val_dataloader)

        self.logger.info(" | ".join([
            f"Fold: {self.fold}",
            f"Test Accuracy: {results['Accuracy']}",
            f"Test AUC: {results['AUC']}",
            f"Test F1: {results['F1']}",
            f"Test Precision: {results['Precision']}",
            f"Test Recall: {results['Recall']}",
            f"Test Specificity: {results['Specificity']}",
            f"Test Sensitivity: {results['Sensitivity']}"
        ]))

        if self.cfg.use_wandb:
            wandb.define_metric("test/Test Accuracy", summary="mean")
            wandb.define_metric("test/Test AUC", summary="mean")
            wandb.define_metric("test/Test F1", summary="mean")
            wandb.define_metric("test/Test Precision", summary="mean")
            wandb.define_metric("test/Test Recall", summary="mean")
            wandb.define_metric("test/Test Specificity", summary="mean")
            wandb.define_metric("test/Test Sensitivity", summary="mean")

            wandb.log({
                "test/Test Accuracy": results['Accuracy'],
                "test/Test AUC": results['AUC'],
                "test/Test F1": {results['F1']},
                "test/Test Precision": results['Precision'],
                "test/Test Recall": results['Recall'],
                "test/Test Specificity": results['Specificity'],
                "test/Test Sensitivity": results['Sensitivity']
            })

        return results
