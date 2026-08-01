import timm
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from datasets import load_dataset
from torchvision import transforms

# 1. Load Food-101 Dataset
dataset = load_dataset("food101")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# 2. Define Model using timm & transformers
# timm provides the optimized ViT architecture
timm_backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=101)

# We use HF ViTForImageClassification for seamless Trainer integration
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", 
    num_labels=101, 
    ignore_mismatched_sizes=True
)

# 3. Preprocessing Function
def preprocess_images(examples):
    images = [img.convert("RGB") for img in examples["image"]]
    inputs = processor(images=images, return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs

tokenized_datasets = dataset.map(preprocess_images, batched=True)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./vit-food101-finetuned",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    logging_steps=100,
    save_strategy="epoch",
    learning_rate=5e-5,
    fp16=True,
)

# 5. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# trainer.train()
# model.save_pretrained("./vit-food101-finetuned")
# processor.save_pretrained("./vit-food101-finetuned")