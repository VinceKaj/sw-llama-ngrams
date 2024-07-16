import gc
from settings import Config
import data
import torch
from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  DataCollatorForLanguageModeling
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

if not Config.DO_TRAINING:
  print("Training is disabled.")
  exit()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
  Config.MODEL_NAME_OR_DIR,
  padding_side="left",
  pad_token="<|eot_id|>"
)

# Load llama model
model = AutoModelForCausalLM.from_pretrained(
  Config.MODEL_NAME_OR_DIR,
  quantization_config=(Config.bnb_config if Config.quantize else None),
  attn_implementation=Config.ATTENTION,
  use_safetensors=True,
  device_map="auto"
)
model.config.use_cache = False

# Freeze base model weights
for param in model.parameters():
  param.requires_grad = False

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# Load adapters
lora_config = LoraConfig(
  r=Config.RANK,
  lora_alpha=Config.ALPHA,
  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  lora_dropout=Config.DROPOUT,
  bias="none",
  task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load data
train_data, _, eval_data = data.get_tokenized_datasets(tokenizer, Config.SW_INSTRUCT_NO_SPEAKERS)

# Training configs
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = SFTConfig(
  output_dir=Config.TRAIN_DIR,
  eval_strategy="epoch",
  per_device_train_batch_size=Config.BATCH_SIZE,
  per_device_eval_batch_size=Config.BATCH_SIZE,
  num_train_epochs=Config.EPOCHS,
  seed=Config.SEED,
  gradient_accumulation_steps=1,
  eval_accumulation_steps=1,
  optim=Config.OPTIMIZER,
  max_seq_length=Config.MAX_LENGTH,
  report_to=Config.REPORT_TO,
)

trainer = SFTTrainer(
  model=model,
  args=training_args,
  data_collator=data_collator,
  train_dataset=train_data,
  eval_dataset=eval_data,
  tokenizer=tokenizer,
  packing=False
)

# TRAIN & SAVE
trainer.train()
trainer.save_model(Config.ADAPTERS_DIR)