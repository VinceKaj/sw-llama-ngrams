from datasets import load_dataset
from settings import Config
from transformers import set_seed
import os

def get_dataset(dir):

  set_seed(Config.SEED)

  data_filenames = [dir + f for f in os.listdir(dir) if f.endswith(".text")]

  # split ratio: 50%/40%/10%
  files_split = {
    "train": data_filenames[:int(len(data_filenames) * 0.5)],
    "prompt": data_filenames[int(len(data_filenames) * 0.5):int(len(data_filenames) * 0.9)],
    "eval": data_filenames[int(len(data_filenames) * 0.9):]
  }
  
  return load_dataset("text", data_files=files_split, sample_by="document")
  

def get_tokenized_datasets(tokenizer, dir):
  def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=False, max_length=Config.MAX_LENGTH)

  dataset = get_dataset(dir)
  tokenized_dataset = dataset.map(tokenize_function, batched=True)

  train_data = tokenized_dataset["train"]
  prompt_data = tokenized_dataset["prompt"]
  eval_data = tokenized_dataset["eval"]

  return train_data, prompt_data, eval_data