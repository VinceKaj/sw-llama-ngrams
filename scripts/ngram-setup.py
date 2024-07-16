import os
from settings import Config
import data
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from tqdm import tqdm
import random

def tokenize_files(file_list):
  global tokenizer

  conversations = load_dataset("text", data_files=file_list, sample_by="document")["train"]["text"]
  tokenized_conversations = []

  for i in tqdm(range(len(conversations))):
    raw_files = conversations[i].split("\n")
    
    # prompt lines
    lines = raw_files[3:7]

    # add generated lines
    for line in raw_files[12:]:
      if line == "": break
      lines.append(line)

    tokenized_conversations.append(
      tokenizer.tokenize('\n'.join(lines))
    )

  return tokenized_conversations 

tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME_OR_DIR, padding_side="left")

print("Tokenizing switchboard prompt dataset...")
sw_files = [f"{Config.SW_NO_SPEAKER_DIR}/{f}" for f in os.listdir(Config.SW_NO_SPEAKER_DIR) if f.endswith(".text")]
prompt_files = sw_files[int(len(sw_files)*0.0):int(len(sw_files)*0.9)]
prompt_tokenized = tokenize_files(prompt_files)

print("Tokenizing base model generated conversations...")
base_gen_files = [f"{Config.BASE_GENERATE_DIR}/{f}" for f in os.listdir(Config.BASE_GENERATE_DIR) if f.endswith(".text")]
base_gen_tokenized = tokenize_files(base_gen_files)

print("Tokenizing tuned model generated conversations...")
trained_gen_files = [f"{Config.TRAINED_GENERATE_DIR}/{f}" for f in os.listdir(Config.TRAINED_GENERATE_DIR) if f.endswith(".text")]
trained_gen_tokenized = tokenize_files(trained_gen_files)

print("Tokenizing test dataset (sw-no-speakers)...")
# filter test dataset
set_seed(42)
random.shuffle(sw_files)
test_files = sw_files[int(len(sw_files) * 0.9):]

test_tokenized = tokenize_files(test_files)

print("Extracting Llama vocabulary...")
vocab = [tokenizer.convert_ids_to_tokens([val])[0] for [key,val] in tokenizer.get_vocab().items()]

try:
  os.mkdir(Config.NGRAM_DATA_DIR)
except:
  pass

print("Writing files...")
with open(f"{Config.NGRAM_DATA_DIR}/base-gen-tokenized.text", "w") as f:
  for conversation in base_gen_tokenized:
    f.write(' '.join(conversation) + '\n')
  
with open(f"{Config.NGRAM_DATA_DIR}/trained-gen-tokenized.text", "w") as f:
  for conversation in trained_gen_tokenized:
    f.write(' '.join(conversation) + '\n')

with open(f"{Config.NGRAM_DATA_DIR}/test-tokenized.text", "w") as f:
  for conversation in test_tokenized:
    f.write(' '.join(conversation) + '\n')

with open(f"{Config.NGRAM_DATA_DIR}/prompt-tokenized.text", "w") as f:
  for conversation in prompt_tokenized:
    f.write(' '.join(conversation) + '\n')

with open(f"{Config.NGRAM_DATA_DIR}/vocab.text", "w") as f:
  f.write('\n'.join(vocab))

print("Done! Ready to train ngrams.")