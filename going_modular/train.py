"""
train a PyTorch image classification model using device agnostic code
"""

import os
import torch

from torchvision import transforms
from timeit import default_timer as timer

import data_setup, model_builder, engine, utils

# setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# setup the directory paths
train_dir = "../Datasets/modular/pizza_steak_sushi/train"
test_dir = "../Datasets/modular/pizza_steak_sushi/test"

# setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
    ])

# Create dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                             test_dir=test_dir,
                                                                             transform=data_transform,
                                                                             batch_size=BATCH_SIZE)

# Create model
torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=HIDDEN_UNITS,
                              output_shape=len(class_names)).to(device)

# Setup loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Start the timer
start_time = timer()

# start training with help from engine.py
engine.train(model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=NUM_EPOCHS,
            device=device)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# save the model to file
utils.save_model(model=model,
                 target_dir="../Models/",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
