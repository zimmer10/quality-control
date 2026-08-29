"""Shared schemas and artifact contracts used by both development tracks."""

OOF_BASE_COLUMNS = ("id", "category", "fold", "group_id", "true_label")

AVAILABILITY_MASK_COLUMNS = (
    "has_ocr",
    "has_embeddings",
    "has_retrieval",
    "has_lora",
)

