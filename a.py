import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "distilgpt2"

DATA_FILE = "data.txt"

OUTPUT_DIR = "./llm-output"

MAX_LENGTH = 512

BATCH_SIZE = 2

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 5e-5

EPOCHS = 3


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using Apple GPU through Metal/MPS")

elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using NVIDIA GPU through CUDA")

else:
    DEVICE = "cpu"
    print("Using CPU")


print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")


# ============================================================
# Load tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT-2 does not have a padding token.
# Use EOS as the padding token.
tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Load dataset
# ============================================================

print("Loading dataset...")

dataset = load_dataset(
    "text",
    data_files={
        "train": DATA_FILE
    }
)

print(f"Original examples: {len(dataset['train'])}")


# ============================================================
# Remove empty examples
# ============================================================

def is_not_empty(example):
    text = example["text"]

    if text is None:
        return False

    return len(text.strip()) > 0


dataset = dataset.filter(is_not_empty)

print(f"Examples after removing empty lines: {len(dataset['train'])}")


# ============================================================
# Tokenization
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )


print("\nTokenizing dataset...")

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)


# ============================================================
# Remove examples that somehow have zero tokens
# ============================================================

def has_tokens(example):
    return len(example["input_ids"]) > 0


tokenized_dataset = tokenized_dataset.filter(has_tokens)

print(
    f"Examples after tokenization: "
    f"{len(tokenized_dataset['train'])}"
)


# ============================================================
# Load model
# ============================================================

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.config.pad_token_id = tokenizer.pad_token_id


# ============================================================
# Data collator
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ============================================================
# Training configuration
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,

    logging_steps=1,

    save_steps=500,

    save_total_limit=2,

    eval_strategy="no",

    # MPS does not use CUDA.
    use_cpu=False,

    remove_unused_columns=True,

    report_to="none",

    # Keep these disabled for the simple MPS setup.
    fp16=False,

    bf16=False,

    # Prevent the MPS pin-memory warning.
    dataloader_pin_memory=False,
)


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset["train"],

    tokenizer=tokenizer,

    data_collator=data_collator,
)


# ============================================================
# Training
# ============================================================

print("\n========================================")
print("Starting training")
print("========================================\n")

trainer.train()


# ============================================================
# Save model
# ============================================================

print("\nSaving model...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)


print("\n========================================")
print("Training complete!")
print(f"Model saved to: {OUTPUT_DIR}")
print("========================================")