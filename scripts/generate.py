from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
)
from peft import PeftModel
from settings import Config
import data
from tqdm import tqdm
import os

def generate_from_prompt(gen_dir_name):
  global tokenizer, model

  try:
    os.mkdir(gen_dir_name)
  except: pass

  prompt_dataset = data.get_dataset(Config.SW_NO_SPEAKER_DIR)["prompt"]

  messages = [
    {
      "role": "user",
      "content": ""
    }
  ]

  PROMPT = "Write a phone conversation between two people. The first few lines of the conversation are:\n"
  terminators = tokenizer.convert_tokens_to_ids(["<|eot_id|>", "\\n"])

  for i in tqdm(range(len(prompt_dataset["text"]))):
    first_four = prompt_dataset["text"][i].split("\n")[:4]

    messages[0]["content"] = PROMPT + "\n".join(first_four) + "\n\nWrite 100 more lines.\n"

    model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

    outputs = model.generate(
      model_inputs,
      max_new_tokens=Config.GENERATION_TOKENS,
      eos_token_id=terminators, 
      repetition_penalty=Config.REPETITION_PENALTY,
      do_sample=Config.USE_SAMPLING 
    )

    response = outputs[0][model_inputs.shape[-1]:]
    with open(f"{gen_dir_name}/gen_{i}.text", "w") as f:
      f.write(tokenizer.decode(model_inputs[0], skip_special_tokens=True))
      f.write(tokenizer.decode(response, skip_special_tokens=True))


tokenizer = AutoTokenizer.from_pretrained(
  Config.MODEL_NAME_OR_DIR,
  padding_side = "left",
  pad_token = "<|eot_id|>"
)

model = AutoModelForCausalLM.from_pretrained(
  Config.MODEL_NAME_OR_DIR,
  use_safetensors=True,
  device_map="auto"
)

if (Config.GENERATE_USING_BASE):
  print("Generating with base model...")

  generate_from_prompt(Config.BASE_GENERATE_DIR)

if (Config.GENERATE_USING_TUNED):
  model = PeftModel.from_pretrained(model, Config.PROMPTS_DIR)
  #model = model.merge_and_unload()

  print("Generating with fine-tuned model...")

  generate_from_prompt(Config.TRAINED_GENERATE_DIR)
