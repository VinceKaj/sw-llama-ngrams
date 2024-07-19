# installation
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
#pip install transformers
#pip install peft accelerate datasets scikit-learn evaluate trl bitsandbytes accelerate tqdm
#conda install cuda -c nvidia -y
#pip install flash-attn --no-build-isolation
# extracting data
#unzip data/sw-instruct-no-speakers.zip -d data
#unzip data/sw-no-speakers.zip -d data
#unzip data/sw-with-speakers.zip -d data
# training
echo "Training llama soft prompts"
python scripts/train.py
echo "Training finished. Evaluating perplexity of fine-tuned model."
# evaluation 
python scripts/evaluate.py
echo "Perplextiy finished (see result above). Generating text for ngram training."
# generation
python scripts/generate.py > llm-perplexities.txt
echo "Generation done. Preparing data for ngram training."
# train ngrams
python scripts/ngram-setup.py
./ngrams.sh
