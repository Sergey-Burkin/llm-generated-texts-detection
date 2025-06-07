import torch
from classifier.model import TextGenerationClassifier
from classifier.utils import prepare_text_for_inference
from omegaconf import DictConfig
from torch.onnx import export
from transformers import AutoTokenizer


def export_to_onnx(cfg: DictConfig, checkpoint_path: str, onnx_path: str):
    # Загрузка модели
    model = TextGenerationClassifier.load_from_checkpoint(
        checkpoint_path, cfg=cfg, num_training_steps=0
    )
    model.eval()  # Переключение в режим вывода

    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    # Подготовка примера входных данных
    sample_text = "This is a sample text for ONNX export."
    inputs = prepare_text_for_inference(
        text=sample_text, tokenizer=tokenizer, max_length=cfg.data.max_length
    )

    # Создание обертки для возврата logits
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model  # Доступ к внутренней модели transformers

        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits

    wrapped_model = ModelWrapper(model)

    # Подготовка входных тензоров с размерностью пакета
    input_ids = inputs["input_ids"].unsqueeze(0)
    attention_mask = inputs["attention_mask"].unsqueeze(0)

    # Динамические оси (динамический размер пакета)
    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "logits": {0: "batch_size"},
    }

    # Экспорт в ONNX с использованием opset версии 14
    export(
        model=wrapped_model,
        args=(input_ids, attention_mask),
        f=onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,  # Обновлено до версии 14
        do_constant_folding=True,
    )

    print(f"Модель успешно экспортирована в {onnx_path}")
