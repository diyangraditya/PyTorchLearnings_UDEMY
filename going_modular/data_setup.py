"""
containes functionality for creating pytorch dataloaders for image classification data
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()
def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
   """
   creating training and testing dataloaders

   takes in a training directory and testing directory path and turns them into pytorch datasets and then into pytorch dataloaders

   Args:
   train_diir: path to training directory.
   test_dir: path to testing directory
   transform: torchvision transforms to perform on training and testing data
   batch_size: number of samples per batch in each of the dataloaders
   num_workers: an integers for number of workers per dataloader

   Returns:
   A tuple of (train_dataloader, test_dataloader, class_names).
   where class_names is a list of the target classes
   example usage:
      train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=path/to/train_dir,
      test_dir=path/to/test_dir,
      transform=some_transform, 
      batch_size=32,
      num_workers=4)
   """
   # use ImageFolder to create datasets
   train_data = datasets.ImageFolder(train_dir, transform=transform)
   test_data = datasets.ImageFolder(test_dir, transform=transform)

   # get class names
   class_names = train_data.classes

   # turn images into DataLoaders
   train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True
   )

   test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True
   )
   return train_dataloader, test_dataloader, class_names
