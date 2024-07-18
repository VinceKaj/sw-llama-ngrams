from transformers import AutoTokenizer, AutoModelForCausalLM
import data
import torch
from peft import PeftModel
from settings import Config
from tqdm import tqdm

# Evaluation

def get_perplexity():

    global tokenizer, model

    max_length = Config.MAX_LENGTH
    stride = Config.STRIDE
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print('Perplexity:', ppl)


tokenizer = AutoTokenizer.from_pretrained(
  Config.MODEL_NAME_OR_DIR,
  padding_side="left",
  pad_token = "<|eot_id|>"
)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    Config.MODEL_NAME_OR_DIR, 
    use_safetensors=True,
    device_map="auto"
)

# Dataset
dataset = data.get_dataset(Config.SW_INSTRUCT_NO_SPEAKERS)
encodings = tokenizer('\n'.join(dataset["eval"]["text"]), padding="max_length", return_tensors='pt', truncation=False, max_length=Config.MAX_LENGTH)

if Config.EVAL_BASE_MODEL:

    print("Evaluating base model...")
    get_perplexity()

if Config.EVAL_TUNED_MODEL:

    print("Loading soft prompts...")
    model = PeftModel.from_pretrained(model, Config.PROMPTS_DIR)
    model = model.merge_and_unload()

    # Fine-tuned model evaluation
    print("Evaluating fine-tuned model...")
    get_perplexity()