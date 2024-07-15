import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Union
from tqdm.auto import tqdm
from ..utils.metrics import ComputeMetrics
import os

def mean(lst):
  return sum(lst) / len(lst)

class EVJVQATrainerMLP:
  """
  EVJVQA trainer class
  Args:
      model ()
  """
  def __init__(self,
               model: nn.Module,
               criterion: nn.Module,
               optimizer: Optimizer,
               scheduler: _LRScheduler,
               device: Union[str, torch.device],
               compute_metrics: ComputeMetrics,
               mixed_precision: bool=False):

    self.model = model
    self.device = device
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.mixed_precision = mixed_precision
    self.compute_metrics = compute_metrics

    if self.mixed_precision:
      self.scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
      self.scaler = None

  def fit(self,
          epochs: int,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          eval_every: int = 1,
          continue_epoch: int = 0):
    """Fitting function to start training and validation of the trainer"""

    for epoch in range(epochs):

      train_scalar_metrics = self.train_epoch(train_dataloader)
      if (epoch + 1) % eval_every == 0:
        val_scalar_metrics = self.val_epoch(val_dataloader)
        states = {
          'epoch': continue_epoch+epoch+1,
          "model_state_dict": self.model.state_dict(),
          "scheduler_state_dict": self.scheduler.state_dict(),
        }
        self.save_checkpoint(continue_epoch+epoch+1, states, 'checkpoint')
        self.save_checkpoint(continue_epoch+epoch+1, train_scalar_metrics, 'train_metrics', metric=True)
        self.save_checkpoint(continue_epoch+epoch+1, val_scalar_metrics, 'val_metrics', metric=True)

      # Log learning rate
      curr_lr = self.optimizer.param_groups[0]['lr']

      if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
        self.scheduler.step(float(val_scalar_metrics["Loss"]))
      else:
        self.scheduler.step()

      new_lr = self.optimizer.param_groups[0]['lr']
      print(f'Old lr: {curr_lr:.6f} - New lr: {new_lr:.6f}')

  def save_checkpoint(self, epoch: int, state: dict, name: str, metric: bool = False):
        
        if metric:
          checkpoint_dir = "metrics"
          checkpoint_name = f"{name}_{epoch}.pth"
          if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        else:
          checkpoint_dir = "checkpoints"
          checkpoint_name = f"{name}_{epoch}.pth"
          if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        filename = checkpoint_dir + '/' + checkpoint_name
        torch.save(state, filename)

  def train_epoch(self, train_dataloader: DataLoader):
    """Training logic for a training epoch"""
    f1 = []
    accuracy = []
    bleu1 = []
    bleu2 = []
    bleu3 = []
    bleu4 = []
    rouge = []
    meteor = []
    loss = []

    self.model.train()
    for batch in tqdm(train_dataloader):

        batch_metrics, pred_examples = self.train_step(batch)

        # Save metrics
        f1.append(batch_metrics['f1'])
        accuracy.append(batch_metrics['accuracy'])
        bleu1.append(batch_metrics['bleu1'])
        bleu2.append(batch_metrics['bleu2'])
        bleu3.append(batch_metrics['bleu3'])
        bleu4.append(batch_metrics['bleu4'])
        rouge.append(batch_metrics['rouge'])
        meteor.append(batch_metrics['meteor'])
        loss.append(batch_metrics['loss'])

        # Print some prediction examples
        print(f"Precdictions: {pred_examples['predictions']}")
        print(f"Ground-Truth: {pred_examples['ground-truth']}")
        print('-'*100)

    scalar_metrics = {
        'Loss': mean(loss),
        'F1': mean(f1),
        'Accuracy': mean(accuracy),
        'Bleu-1': mean(bleu1),
        'Bleu-2': mean(bleu2),
        'Bleu-3': mean(bleu3),
        'Bleu-4': mean(bleu4),
        'Rouge': mean(rouge),
        'Meteor': mean(meteor),
    }
    print('Traing epoch stats: ')
    print(scalar_metrics)
    return scalar_metrics

  def train_step(self, batch):

    # Move data to device
    batch = batch.to(self.device)

    if self.mixed_precision:
      with torch.autocast(device_type="cuda", dtype=torch.float16):

        # Forward pass
        outputs = self.model(batch) # predictions (`torch.Tensor` with shape `batch_size, seq_len, vocab_size`)

        # Calculate loss
        loss = self.criterion(outputs.mT, batch.answers_tokenized.input_ids)

        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.model.zero_grad()

    else:
      # Forward pass
        outputs = self.model(batch) # predictions (`torch.Tensor` with shape `batch_size, seq_len, vocab_size`)

        # Calculate loss
        loss = self.criterion(outputs.mT, batch.answers_tokenized.input_ids)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.zero_grad()

    batch_metrics, pred_examples = self.compute_metrics.compute(outputs[0].argmax(dim=-1), batch.answers_tokenized)
    batch_metrics['loss'] = loss.item()

    return batch_metrics, pred_examples

  def val_epoch(self, val_dataloader: DataLoader):
    """Evaluate logic for a validation epoch"""
    f1 = []
    accuracy = []
    bleu1 = []
    bleu2 = []
    bleu3 = []
    bleu4 = []
    rouge = []
    meteor = []
    loss = []
    aux_loss = []
    total_loss = []

    self.model.eval()
    with torch.no_grad():
      for batch in tqdm(val_dataloader):

          batch_metrics, pred_examples = self.val_step(batch)

          # Save metrics
          f1.append(batch_metrics['f1'])
          accuracy.append(batch_metrics['accuracy'])
          bleu1.append(batch_metrics['bleu1'])
          bleu2.append(batch_metrics['bleu2'])
          bleu3.append(batch_metrics['bleu3'])
          bleu4.append(batch_metrics['bleu4'])
          rouge.append(batch_metrics['rouge'])
          meteor.append(batch_metrics['meteor'])
          loss.append(batch_metrics['loss'])

          # Print some prediction examples
          print(f"Precdictions: {pred_examples['predictions']}")
          print(f"Ground-Truth: {pred_examples['ground-truth']}")
          print('-'*100)

    scalar_metrics = {
        'Loss': mean(loss),
        'F1': mean(f1),
        'Accuracy': mean(accuracy),
        'Bleu-1': mean(bleu1),
        'Bleu-2': mean(bleu2),
        'Bleu-3': mean(bleu3),
        'Bleu-4': mean(bleu4),
        'Rouge': mean(rouge),
        'Meteor': mean(meteor),
    }
    print('Evaluate epoch stats: ')
    print(scalar_metrics)
    return scalar_metrics

  def val_step(self, batch):

    # Move data to device
    batch = batch.to(self.device)

    if self.mixed_precision:
      with torch.autocast(device_type="cuda", dtype=torch.float16):

        # Forward pass
        outputs = self.model(batch) # predictions (`torch.Tensor` with shape `batch_size, seq_len, vocab_size`)

        # Calculate loss
        loss = self.criterion(outputs.mT, batch.answers_tokenized.input_ids)

    else:
      # Forward pass
        outputs = self.model(batch) # predictions (`torch.Tensor` with shape `batch_size, seq_len, vocab_size`)

        # Calculate loss
        loss = self.criterion(outputs.mT, batch.answers_tokenized.input_ids)

    batch_metrics, pred_examples = self.compute_metrics.compute(outputs[0].argmax(dim=-1), batch.answers_tokenized)
    batch_metrics['loss'] = loss.item()

    return batch_metrics, pred_examples