import gc
from settings import Config
import data
import torch
from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  DataCollatorForLanguageModeling,
  TrainingArguments,
  Trainer
)
from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit

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

# Load soft prompts
propmts_config = PromptTuningConfig(
  task_type=TaskType.CAUSAL_LM,
  prompt_tuning_init=PromptTuningInit.RANDOM,
  num_virtual_tokens=Config.VIRTUAL_TOKENS,
  prompt_tuning_init_text=Config.PROMPT_TEXT,
  tokenizer_name_or_path=Config.MODEL_NAME_OR_DIR
)
model = get_peft_model(model, propmts_config)
model.print_trainable_parameters()

# Load data
train_data, _, eval_data = data.get_tokenized_datasets(tokenizer, Config.SW_INSTRUCT_NO_SPEAKERS)

def create_training_arguments(path, learning_rate=0.0035, epochs=6):
  return TrainingArguments(
    output_dir=path,
    auto_find_batch_size=True,
    learning_rate=learning_rate,
    num_train_epochs=epochs,
  )

def create_trainer(model, training_args, train_dataset):
  return Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), # mlm meas masked language modeling
    train_dataset=train_dataset
  )

training_args_prompt = create_training_arguments(Config.TRAIN_DIR, learning_rate=0.0035, epochs=Config.EPOCHS)
trainer = create_trainer(model, training_args_prompt, train_data)
trainer.train()

trainer.model.save_pretrained(Config.PROMPTS_DIR)
