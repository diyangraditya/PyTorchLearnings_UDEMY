
import argparse
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms
import data_setup, engine, model_builder, utils
import data_setup
import engine
import model_builder
import utils


parser = argparse.ArgumentParser(description="Train a TinyVGG image classifier.")
parser.add_argument(
    "--train_dir",
    type=str,
    default="../Datasets/modular/pizza_steak_sushi/train",
    help="Directory containing training images.",
)
parser.add_argument(
    "--test_dir",
    type=str,
    default="../Datasets/modular/pizza_steak_sushi/test",
    help="Directory containing testing images.",
)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--hidden_units", type=int, default=10)
args = parser.parse_args()


def resolve_data_dir(path):
    """Support data extracted either directly or into a nested folder."""
    path = Path(path)
    if path.is_dir():
        return path

    nested_path = path.parent / "pizza_steak_sushi" / path.name
    if nested_path.is_dir():
        return nested_path

    raise FileNotFoundError(f"Could not find data directory: {path}")


train_dir = resolve_data_dir(args.train_dir)
test_dir = resolve_data_dir(args.test_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

image_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=image_transform,
    batch_size=args.batch_size,
)

model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=args.hidden_units,
    output_shape=len(class_names),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=args.num_epochs,
    device=device,
)

utils.save_model(
    model=model,
    target_dir="../Models",
    model_name="05_going_modular_script_mode_tinyvgg_model.pth",
)
