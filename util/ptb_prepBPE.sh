#!/bin/bash 

num_merges=$1
train=$2
valid=$3
test=$4

dir=data
if [ ! -d $dir ]; then
  mkdir -p $dir
fi
codes=$dir/ptb_codes$num_merges

cat $train $valid | ./learn_bpe.py -s $num_merges -o $codes

./apply_bpe.py -c $codes < $train  | sed -r 's/@@//g' > $dir/ptb_train${num_merges}BPE.txt
./apply_bpe.py -c $codes < $valid  | sed -r 's/@@//g' > $dir/ptb_valid${num_merges}BPE.txt
./apply_bpe.py -c $codes < $test  | sed -r 's/@@//g' > $dir/ptb_test${num_merges}BPE.txt

mv $dir/ptb_train${num_merges}BPE.txt ../data/ptb/train${num_merges}BPE.txt
mv $dir/ptb_valid${num_merges}BPE.txt ../data/ptb/valid${num_merges}BPE.txt
mv $dir/ptb_test${num_merges}BPE.txt ../data/ptb/test${num_merges}BPE.txt
