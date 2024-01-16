import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from models import UNet
from utils.management import load_config_from_run, load_model_from_path
from utils.logging import print_

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
        self.device = device
        self.model = UNet(n_channels=3, n_classes=30)
        # Initialize the Data
        self.load_data()
        return
    
    ##-- Initialization Functions --##
    def load_data(self, train:bool=True, eval:bool=True):
        """ Load the training and evaluation datasets. """

        ##-- Train Dataset --##
        if train:
            train_dataset = datasets.Cityscapes(
                root='./data/cityscapes',
                split='train',
                mode='fine',
                target_type='semantic',
                download=True,
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
                download=True,
                transform=None
            )
            self.eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size = self.config['batch_size'],
                shuffle=False,
                num_workers=4,
                drop_last=True
            )
    def initialize_model(self, checkpoint_path:str=None):
        self.model = UNet(n_channels=3, n_classes=30)
        if checkpoint_path:
            load_model_from_path(checkpoint_path, self.model, device=self.device)

    def initialize_optimization(self):
        assert self.model is not None, "Please initialize the model before initializing the optimizer."
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config['scheduler_factor'],
            patience=self.config['scheduler_patience']
        )

    ##-- Training Functions --##
    def train(self):
        
        max_epochs = self.config['num_epochs']

        for epoch in range(max_epochs):
            print_(f'Epoch: {epoch+1}/{max_epochs}')

            if epoch % self.config['eval_interval'] == 0:
                self.eval_epoch()

    def train_epoch(self):
        return
    
    ##-- Evaluation Functions --##
                
    def eval_epoch(self):
        return

