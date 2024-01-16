import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from models import UNet
from utils.management import load_config_from_run, load_model_from_path
from utils.logging import print_, Logger, MetricTracker, LOGGER

class Trainer(object):

    def __init__(self,
                 experiment_name: str,
                 run_name: str,
                 device:str='cpu'):
        
        self.config = load_config_from_run(experiment_name, run_name)
        ##-- Modules --##
        self.model = None
        self.train_loader = None
        self.eval_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.logger = None
        self.model = None
        ##-- Parameters --##
        self.device = device
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.step = 1
        return
    
    ##-- Initialization Functions --##
    def initialize_data(self, train:bool=True, eval:bool=True):
        """ Load the training and evaluation datasets. """

        ##-- Train Dataset --##
        if train:
            train_dataset = datasets.Cityscapes(
                root='./data/cityscapes',
                split='train',
                mode='fine',
                target_type='semantic',
                transform=None
            )
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_size = self.config['batch_size'],
                shuffle=True,
                num_workers=4,
                drop_last=True                               
            )

        ##-- Evaluation Dataset --##
        if eval:
            eval_dataset = datasets.Cityscapes(
                root='./data/cityscapes',
                split='val',
                mode='coarse',
                transform=None
            )
            self.eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size = self.config['batch_size'],
                shuffle=True,
                num_workers=4,
                drop_last=True
            )
    def initialize_model(self, checkpoint_path:str=None):
        self.model = UNet(n_channels=3, n_classes=30)
        if checkpoint_path:
            load_model_from_path(checkpoint_path, self.model, device=self.device)
        self.logger.log_architecture(self.model)

    def initialize_optimization(self):
        """ Initialize the optimization for training."""
        assert self.model is not None, "Please initialize the model before initializing the optimizer."
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config['scheduler_factor'],
            patience=self.config['scheduler_patience']
        )
        self.criterion = nn.CrossEntropyLoss()

    def initialize_logging(self):
        """
            For demonstration sake we here initialize the full bandwidth of logging.
        """
        if LOGGER is None:
            self.logger = Logger(self.experiment_name, self.run_name)
        # Initialize the tensorboard logging
        self.logger.initialize_tensorboard()
        # Initialize the WandB logging
        self.logger.initialize_wandb(project_name="ExperimentTemplate")
        # Initialize the logging to a CSV file
        self.logger.initialize_csv()
        # Initialize the metric tracker to to track the evaluation metrics
        self.metrics = MetricTracker()

    ##-- Training Functions --##
    def train(self):
        """ Train the model. """
        assert self.optimizer is not None and self.scheduler is not None and self.criterion is not None, "Optimization not properly initialized."
        assert self.model is not None, "Please initialize the model before training."
        assert self.train_loader is not None, "Please initialize the train loader before training."
        assert self.eval_loader is not None, "Please initialize the eval loader before training."
        
        max_epochs = self.config['num_epochs']

        for epoch in range(max_epochs):
            print_(f'Epoch: {epoch+1}/{max_epochs}')
            # Evaluate the model in regular intervals
            if epoch==0 or (epoch+1) % self.config['evaluation_frequency'] == 0:
                self.eval_epoch()
            # Train the model
            self.train_epoch()
            # Save a model checkpoint in regular intervals
            if (epoch+1) % self.config['eval_interval'] == 0:
                self.logger.save_checkpoint(
                    epoch=epoch+1,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler
                )

    def train_epoch(self):

        running_loss = 0.0

        length = len(self.train_loader)
        progress_bar = tqdm(enumerate(self.train_loader), total=length)

        # Initialize the training
        self.model.train()
        self.metrics.reset()

        for i, (img, target) in progress_bar:
            if i == length:
                break
            img = img.to(self.device)
            target = target.to(self.device)
            target = F.one_hot(target, num_classes=30)

            pred = self.model(img)

            if torch.isnan(pred).any():
                print_('\nNaN in prediction', 'warning')
                continue
            # Optimization process
            loss = self.criterion(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss = 0.8*running_loss + 0.2*loss.item()
            progress_bar.set_description(f'Loss: {running_loss:.4f}')
            # Log the training loss
            self.logger.log({'CrossEntropy': loss.item()}, step=self.step)




    ##-- Evaluation Functions --##
    @torch.no_grad()      
    def eval_epoch(self):

        print_('\nEvaluation:')
        
        length = len(self.eval_loader)
        progress_bar = tqdm(enumerate(self.eval_loader), total=length)
        # Initialize the evaluation
        self.model.eval()
        self.metrics.reset()

        for i, (img, target) in progress_bar:
            if i == length:
                break
            img = img.to(self.device)
            target = target.to(self.device)
            pred = self.model(img)
            # Compute the evaluation metrics
            target = F.one_hot(target, num_classes=30)
            self.evaluation_metrics(pred, target)

            # Visualize some demonstration images for the first batch of images
 
        # Retrieve the averages over all iterations
        accuracy = self.metrics.get_average('accuracy')
        iou = self.metrics.get_average('iou')
        precision = self.metrics.get_average('precision')
        recall = self.metrics.get_average('recall')
        dice = self.metrics.get_average('dice')
        print_('\nEvaluation Results:')
        print_(f'  Accuracy: \t{accuracy:.4f}')
        print_(f'  IoU: \t{iou:.4f}')
        print_(f'  Precision: \t{precision:.4f}')
        print_(f'  Recall: \t{recall:.4f}')
        print_(f'  Dice: \t{dice:.4f}')
        # Log the evaluation metrics
        self.logger.log({
            'accuracy': accuracy,
            'iou': iou,
            'precision': precision,
            'dice': dice,
            'recall': recall,
        }, self.step)


    @torch.no_grad()      
    def evaluation_metrics(self, segmentation:torch.Tensor, target:torch.Tensor):
        """
            Compute some evaluation metrics for semantic segmentation for each class.
            We assume that the segmentation and ground truth is given as a one-hot encoded tensor.
            
        """


        class_acc = []
        class_dice = []
        class_prec = []
        class_recall = []
        class_iou = []
        # Compute per class metrices
        for id in range(30):
            class_pred = segmentation[:,id]
            class_gt = target[:,id]

            seg_flat = torch.flatten(class_pred)
            target_flat = torch.flatten(class_gt)
            # Compute the confusion matrix per class
            tp = torch.sum(target_flat * seg_flat)
            fp = torch.sum(seg_flat) - tp
            fn = torch.sum(target_flat) - tp
            tn = seg_flat.numel() - tp - fp - fn
            # Compute the quantitative metrics
            acc = (tn + tp) / (tn + tp + fp + fn)
            prec = tp / (tp + fp)
            recall = tp / (tp + fn)
            dice = (2*tp) / (2*tp+fp+fn)
            iou = tp / (tp + fp + fn)

            class_acc.append(acc)
            class_dice.append(dice)
            class_prec.append(prec)
            class_recall.append(recall)
            class_iou.append(iou)
        # Compute the averages over the per-class metrices
        acc_avg = np.mean(class_acc)
        dice_avg = np.mean(class_dice)
        prec_avg = np.mean(class_prec)
        recall_avg = np.mean(class_recall)
        iou_avg = np.mean(class_iou)
        # Update the metric tracker to save metrices intermediately without polluting the RAM
        self.metrics.update(name='class_acc', val=class_acc)
        self.metrics.update(name='class_dice', val=class_dice)
        self.metrics.update(name='class_prec', val=class_prec)
        self.metrics.update(name='class_recall', val=class_recall)
        self.metrics.update(name='class_iou', val=class_iou)
        self.metrics.update(name='accuracy', val=acc_avg)
        self.metrics.update(name='dice', val=dice_avg)
        self.metrics.update(name='precision', val=prec_avg)
        self.metrics.update(name='recall', val=recall_avg)
        self.metrics.update(name='iou', val=iou_avg)


