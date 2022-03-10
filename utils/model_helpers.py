import pytorch_lightning as pl
import torch
from torch import nn
from IPython import display
import matplotlib.pylab as plt

class DeepLearningModel(pl.LightningModule):
    '''
    PyTorch Lightning class for a deep learning model
    '''
    def __init__(self, imheight=28, imwidth=28, print_shapes=False, plot_progress=None):
        super().__init__()
        
        self.print_shapes = print_shapes
        
        self.plot_progress = plot_progress
        
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        
        '''This function defines what happens when we feed an image through the network'''
        
        pass

    
    def configure_optimizers(self):
        '''
        This method defines the optimizer used to train the model.
        '''
        # To do this, we first need to set up a few variables:

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # This optimizer uses Adam and has learning rate of 0.001. We will cover this more in depth later!
        # You can play around with this!
        
        return optimizer

    
    def training_step(self, batch, batch_idx):
        '''
        This method defines what happens each step of training.
        '''
        
        x, y = batch
        y_hat = self(x) # We feed foward the input
        train_loss = self.loss_function(y_hat, y) # We compute the loss
        
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        '''
        This method defines what happens each step of the validation.
        '''
        
        x, y = batch
        y_hat = self(x) # We feed foward the input
        num_correct = y_hat.argmax(dim=1).eq(y).sum().item() # This counts the number of correct predictions
        num_total = len(x) # The total number in the batch  
        
        validation_return_dict = {'num_correct': correct, 'num_total': total}
        
        return validation_return_dict

    def validation_epoch_end(self, outputs):
        '''
        This method defines what happens after all the validation steps are run.
        '''
        
        num_correct = sum([x["num_correct"] for  x in outputs])
        num_total = sum([x["num_total"] for  x in outputs])
        acc = num_correct / num_total
        
        if self.plot_progress is not None:
            self.plot_progress.update(acc)
            
        self.log('val_acc', num_correct/num_total)