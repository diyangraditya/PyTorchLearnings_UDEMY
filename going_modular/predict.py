
"""
Predicts a single image with a trained TinyVGG model.

Example usage (from the Notebooks directory):
    python ../going_modular/predict.py --image ../Datasets/modular/pizza_steak_sushi/test/sushi/1172255.jpg
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import model_builder


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for prediction."""
    parser = argparse.ArgumentParser(description="Predict a class for a single image.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image to predict on.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../Models/05_going_modular_script_mode_tinyvgg_model.pth",
        help="Path to a saved model state_dict (.pth).",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="../Datasets/modular/pizza_steak_sushi/test",
        help="Directory used to infer class names (subfolder names, sorted).",
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=128,
        help="Hidden units the model was trained with (must match training).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run prediction on ('cuda' or 'cpu').",
    )
    return parser.parse_args()


def get_class_names(test_dir: str) -> list[str]:
    """Recover class names from a dataset directory's subfolders.

    ImageFolder sorts class names alphabetically during training, so
    sorting here keeps the same label order the model learned.
    """
    class_names = sorted(
        entry.name for entry in Path(test_dir).iterdir() if entry.is_dir()
    )
    if not class_names:
        raise FileNotFoundError(f"No class subfolders found in: {test_dir}")
    return class_names


def load_model(
    model_path: str,
    hidden_units: int,
    class_names: list[str],
    device: torch.device,
):
    """Rebuild the TinyVGG architecture and load trained weights into it."""
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=hidden_units,
        output_shape=len(class_names),
    ).to(device)

    # map_location lets the checkpoint load on CPU-only machines too
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(
    model,
    image_path: str,
    class_names: list[str],
    device: torch.device,
):
    """Make a prediction on a single image. Returns (class_name, probability)."""
    # Keep this transform identical to the one used in train.py
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    # PIL + ToTensor matches how ImageFolder loaded the training images
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # add batch dim

    with torch.inference_mode():
        pred_logits = model(image_tensor)
        pred_probs = torch.softmax(pred_logits, dim=1)
        pred_label_idx = pred_probs.argmax(dim=1).item()

    return class_names[pred_label_idx], pred_probs[0, pred_label_idx].item()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    class_names = get_class_names(args.test_dir)
    model = load_model(
        model_path=args.model_path,
        hidden_units=args.hidden_units,
        class_names=class_names,
        device=device,
    )

    pred_class, pred_prob = predict(
        model=model,
        image_path=args.image,
        class_names=class_names,
        device=device,
    )

    print(f"[INFO] Image: {args.image}")
    print(f"[INFO] Pred class: {pred_class} | Prob: {pred_prob:.3f}")


if __name__ == "__main__":
    main()
