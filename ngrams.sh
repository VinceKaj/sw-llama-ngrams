models=("prompt" "base-gen" "trained-gen")
n="7"
for value in "${models[@]}"
do
  # train
  echo "Training & evaluting ngram token model with n = $n ($value dataset)"
  cat data/ngrams/$value-tokenized.text | /disk/data2/s1569734/software/kaldi/tools/srilm/bin/i686-m64/ngram-count -kndiscount -interpolate -sort -text - -order $n -lm data/ngrams/token-model-$n-$value.lm
  # evaluate perplexity
  echo "Perplexity of ngram token model with n = $n ($value dataset):" >> llm-perplexities.txt
  /disk/data2/s1569734/software/kaldi/tools/srilm/bin/i686-m64/ngram -lm data/ngrams/token-model-$n-$value.lm -order $n -ppl data/ngrams/test-tokenized.text \
  | awk '/ppl= / {print $6}' >> llm-perplexities.txt
done