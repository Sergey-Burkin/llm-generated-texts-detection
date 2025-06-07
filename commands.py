import fire
import torch
from classifier.model import TextGenerationClassifier
from classifier.packaging import export_to_onnx
from classifier.train import train
from classifier.utils import prepare_text_for_inference
from hydra import compose, initialize
from transformers import AutoTokenizer


def train_model(config_name: str = "config", config_path: str = "./configs"):
    """Train the model using Hydra configuration"""
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)
        train(cfg)


def infer(text: str, config_name: str = "config", config_path: str = "./configs"):
    """Run inference on a sample text"""
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
        model = TextGenerationClassifier.load_from_checkpoint(
            cfg.inference.checkpoint_path,
            cfg=cfg,
            num_training_steps=0,  # Not used for inference
        )
        model.eval()

        # Prepare input
        inputs = prepare_text_for_inference(
            text, tokenizer, max_length=cfg.data.max_length
        )

        # Run inference
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].unsqueeze(0).to(device)
        attention_mask = inputs["attention_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            prob = torch.softmax(logits, dim=1)[0][pred].item()

        label = "Generated" if pred == 1 else "Human-written"
        print(f"Prediction: {label} (confidence: {prob:.2f})")


def export_onnx(config_name: str = "config", config_path: str = "./configs"):
    """Экспортирует модель в формат ONNX"""
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)
        export_to_onnx(
            cfg=cfg,
            checkpoint_path=cfg.inference.checkpoint_path,
            onnx_path=cfg.onnx_export.model_path,
        )


if __name__ == "__main__":
    fire.Fire(
        {
            "train": train_model,
            "infer": infer,
            "export_onnx": export_onnx,
        }
    )
