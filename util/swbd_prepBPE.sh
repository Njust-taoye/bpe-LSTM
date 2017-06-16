#!/bin/bash 

num_merges=$1
train=$2
valid=$3

dir=data
if [ ! -d $dir ]; then
  mkdir -p $dir
fi
codes=$dir/swbd_codes$num_merges

cat $train $valid | ./learn_bpe.py -s $num_merges -o $codes

./apply_bpe.py -c $codes < $train  | sed -r 's/@@//g' > $dir/swbd_train${num_merges}BPE.txt
./apply_bpe.py -c $codes < $valid  | sed -r 's/@@//g' > $dir/swbd_valid${num_merges}BPE.txt

mv $dir/swbd_train${num_merges}BPE.txt ../data/swbd/train${num_merges}BPE.txt
mv $dir/swbd_valid${num_merges}BPE.txt ../data/swbd/valid${num_merges}BPE.txt
