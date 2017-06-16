# large model
CUDA_VISIBLE_DEVICES=`/home/keli1/free-gpu` python train.py --savefile bpe-v1000-he650-batch20 --highway_layers 0 --predict_bpes 0 --predict_words 1 --max_epochs 25 --rnn_size 650 --bpe_vec_size 650 --n_bpes 1000 --n_words 30000 --batch_size 20 --data_dir data/swbd --model_dir cv_swbd_words
CUDA_VISIBLE_DEVICES=`/home/keli1/free-gpu` python train.py --savefile bpe-v1000-he650-batch20 --highway_layers 0 --predict_bpes 1 --predict_words 0 --max_epochs 25 --rnn_size 650 --bpe_vec_size 650 --n_bpes 1000 --n_words 30000 --batch_size 20 --data_dir data/swbd --model_dir cv_swbd_bpes
# small model
CUDA_VISIBLE_DEVICES=`/home/keli1/free-gpu` python train.py --savefile bpe-v1000-he200-batch20 --highway_layers 0 --predict_bpes 0 --predict_words 1 --max_epochs 25 --rnn_size 200 --bpe_vec_size 200 --n_bpes 1000 --n_words 30000 --batch_size 20 --data_dir data/swbd --model_dir cv_swbd_words
CUDA_VISIBLE_DEVICES=`/home/keli1/free-gpu` python train.py --savefile bpe-v1000-he200-batch20 --highway_layers 0 --predict_bpes 1 --predict_words 0 --max_epochs 25 --rnn_size 200 --bpe_vec_size 200 --n_bpes 1000 --n_words 30000 --batch_size 20 --data_dir data/swbd --model_dir cv_swbd_bpes
