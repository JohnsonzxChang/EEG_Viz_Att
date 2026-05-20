import argparse

import torch

from contrast.clips import CLIPContrastiveLoss, HFCLIPEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HFCLIPEncoder(model_name=args.model, device=device).to(device)
    model.eval()

    images = torch.rand(4, 3, 224, 224)
    texts = ["a photo of a cat", "a photo of a dog", "a photo of a car", "a photo of a tree"]

    out = model(images=list(images), texts=texts)
    loss_fn = CLIPContrastiveLoss(label_smoothing=0.0)
    loss = loss_fn(out.logits_per_image, out.logits_per_text)

    print("logits_per_image:", tuple(out.logits_per_image.shape))
    print("logits_per_text:", tuple(out.logits_per_text.shape))
    print("loss:", float(loss.item()))


if __name__ == "__main__":
    main()
