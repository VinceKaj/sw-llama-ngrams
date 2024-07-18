# sw-llama-ngrams
This repository offers scripts and documentation for how to fine tune an LLM (Llama-3-8B-Instruct) on switchboard data using LoRa adapters. The fine-tuned models is used to generate conversations in the style of switchboard and training ngram models to evaluate their perplexity.

# Usage

1. Clone the repository
2. Setup a conda environment with python=3.11 and activate it.
3. Download switchboard-1 corpus and place in it in data/switchboard. Record the exact name directory in [`settings.py`](scripts/settings.py)
4. Run the [switchboard-processing.ipynb](switchboard-processing.ipynb) to format switchboard switchboard for fine-tuning and prompting
5. Clone the [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) repository from huggingface. You will first need to apply for access through huggingface. Save the path to it in [`settings.py`](scripts/settings.py) under `MODEL_NAME_OR_DIR`.
6. Tweak [`scripts/settings.py`](scripts/settings.py) to desired specifications, or keep defaults
7. Run [`main.sh`](main.sh) to fine-tune Llama, evaluate perplexity, run inference and train ngram models. 
8. Find the perplexity of Llama (base and fine-tuned) and ngrams (trained on 3 separate datasets) in `llm-perplexities.txt`

## Explanation of directories and scripts

* `/scripts` contains the necessary python scripts and methods for fine-tuning, LLM perplexity evaluation, and text generation.
* `/data` should contain a switchboard directory, so [switchboard-processing.ipynb](switchboard-processing.ipynb) can find it and process it accordingly. All other files and subdirectories will be created at runtime (see the [settings](scripts/settings.py) script for more details).

# Performance metrics

## Training

* An unquantized model with rank=64, batch size=2 requires 53.2 GB of GPU memory to train. This completes in ~4 hours.

## Generation

When generating, the model is given a hard-coded prompt and 4 lines of conversation from switchboard, then is asked to continue the conversation.

* An unquantized model takes around 4 hours to generate 975 conversations (EIDF with 92 CPU cores and 80 GB GPU memory).

## Perplexity

* Base Llama-3-8B-Instruct perplexity when prompted to generate switchboard conversations is 19.01
* The fine-tuned model with R=64 has a perplexity of 12.61

Ngram perplexities for n=7 are as follows (trained on 975 conversations, 40% of switchboard):

* Base ngram perplexity trained on switchboard prompt dataset is 50.62246
* Ngram trained on data generated by Base Llama-3-8B-Instruct had a perplexity of 111.3882
* Ngarm model trained on data generated by fine-tuned Llama with R=64 had a perplexity of 84.52671
