import torch
from classifier.model import TextGenerationClassifier
from classifier.utils import prepare_text_for_inference
from omegaconf import DictConfig
from torch.onnx import export
from transformers import AutoTokenizer


def export_to_onnx(cfg: DictConfig, checkpoint_path: str, onnx_path: str):
    onnx_cfg = cfg.onnx_export

    model = TextGenerationClassifier.load_from_checkpoint(
        checkpoint_path, cfg=cfg, num_training_steps=0
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    sample_text = "This is a sample text for ONNX export."
    inputs = prepare_text_for_inference(
        text=sample_text, tokenizer=tokenizer, max_length=cfg.data.max_length
    )

    # Обертка модели для возврата только логитов
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model

        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits

    wrapped_model = ModelWrapper(model)

    input_ids = inputs["input_ids"].unsqueeze(0)
    attention_mask = inputs["attention_mask"].unsqueeze(0)

    dynamic_axes = {}
    for tensor_name, axes in onnx_cfg.dynamic_axes.items():
        dynamic_axes[tensor_name] = {axis_idx: f"dim_{axis_idx}" for axis_idx in axes}

    export(
        model=wrapped_model,
        args=(input_ids, attention_mask),
        f=onnx_cfg.model_path,
        input_names=onnx_cfg.input_names,
        output_names=onnx_cfg.output_names,
        dynamic_axes=dynamic_axes,
        opset_version=onnx_cfg.opset_version,
        do_constant_folding=onnx_cfg.do_constant_folding,
        verbose=onnx_cfg.verbose,
    )

    print(f"ONNX модель успешно экспортирована в: {onnx_cfg.model_path}")
