from transformers import AutoTokenizer


def prepare_text_for_inference(
    text: str, tokenizer: AutoTokenizer, max_length: int = 256
):
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"].flatten(),
        "attention_mask": encoding["attention_mask"].flatten(),
    }
