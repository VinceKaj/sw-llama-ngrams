rm -r data/ngrams/models
mkdir data/ngrams/models

echo "n,dataset,conversations,perplexity" > ngram-perplexities.csv

models=("prompt" "base-gen" "trained-gen")
conversations=("50" "100" "200" "300" "400" "500" "600" "700" "800" "900" "975" "1000" "1100" "1200" "1300" "1400" "1500" "1600" "1700" "1800" "1900" "2000" "2100" "2194")
n="7"
for count in "${conversations[@]}"
do
  for value in "${models[@]}"
  do
    # train
    echo "Training & evaluting ngram token model with n = $n ($value dataset) ($count conversations)"
    cat data/ngrams/train/$value-tokenized-$count.text | ngram-count -kndiscount -interpolate -sort -text - -order $n -lm data/ngrams/models/token-model-$n-$value-$count.lm
    # evaluate perplexity
    ngram -lm data/ngrams/models/token-model-$n-$value-$count.lm -order $n -ppl data/ngrams/test-tokenized.text \
    | echo $n,$value,$count,$(awk '/ppl= / {print $6}') >> ngram-perplexities.csv
  done
done
