from __future__ import print_function

import argparse
import json
import numpy as np
import os
import cPickle as pickle
from util.BatchLoaderBPE import BatchLoaderBPE, Tokens
from util.BatchLoaderWord import BatchLoaderWord, Tokens
from model.BPELSTM import BPELSTM, load_model
from math import exp

Train, Validation, Test = 0, 1, 2

def main(opt):
    # for prediction on BPE level
    if opt.predict_bpes:
        loader = BatchLoaderBPE(opt.tokens, opt.data_dir, opt.batch_size, opt.seq_length, opt.n_bpes, opt.ngram)
    # for prediction on Word level
    if opt.predict_words:
        loader = BatchLoaderWord(opt.tokens, opt.data_dir, opt.batch_size, opt.seq_length, opt.n_bpes, opt.n_words, opt.ngram)
        opt.word_vocab_size = min(opt.n_words, len(loader.idx2word))
        print('Word vocab size: ', opt.word_vocab_size)
    opt.bpe_vocab_size = min(opt.n_bpes, len(loader.idx2bpe))
    print('BPE vocab size: ', opt.bpe_vocab_size)
    
    # define the model
    if not opt.skip_train:
        print('creating an BPELSTM with ', opt.num_layers, ' layers')
        model = BPELSTM(opt)
        # make sure output directory exists
        if not os.path.exists(opt.model_dir):
            os.makedirs(opt.model_dir)
        pickle.dump(opt, open('{}/{}.pkl'.format(opt.model_dir, opt.savefile), "wb"))
        model.save('{}/{}.json'.format(opt.model_dir, opt.savefile))
        logfile=open(opt.model_dir + '/' + opt.savefile, 'w')
        model.fit_generator(loader.next_batch(Train), loader.split_sizes[Train], opt.max_epochs,
                            loader.sub_batch(Train), loader.split_sizes[Train]/10,
                            loader.next_batch(Validation), loader.split_sizes[Validation], opt, logfile)
        logfile.close()
        model.save_weights('{}/{}.h5'.format(opt.model_dir, opt.savefile), overwrite=True)
    else:
        model = load_model('{}/{}.json'.format(opt.model_dir, opt.savefile))
        model.load_weights('{}/{}.h5'.format(opt.model_dir, opt.savefile))
        print(model.summary())

    # evaluate on full test set.
    # only PTB has test set
    if opt.data_dir == "data/ptb":
        test_perp = model.evaluate_generator(loader.next_batch(Test), loader.split_sizes[Test])
        print('Perplexity on test set: ', exp(test_perp))

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train a word+character-level language model')
    # data
    parser.add_argument('--data_dir', type=str, default='data/ptb', help='data directory. Should contain train.txt/valid.txt/test.txt with input data')
    # model params
    parser.add_argument('--predict_bpes', type=int, default=0, help='predict on bpes (1=yes)')
    parser.add_argument('--predict_words', type=int, default=1, help='predict on words (1=yes)')
    parser.add_argument('--rnn_size', type=int, default=200, help='size of LSTM internal state')
    parser.add_argument('--highway_layers', type=int, default=2, help='number of highway layers')
    parser.add_argument('--word_vec_size', type=int, default=200, help='dimensionality of word embeddings')
    parser.add_argument('--bpe_vec_size', type=int, default=200, help='dimensionality of bpes embeddings')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the LSTM')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout. 0 = no dropout')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=1, help='starting learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.5, help='learning rate decay')
    parser.add_argument('--decay_when', type=float, default=1, help='decay if validation perplexity does not improve by more than this much')
    parser.add_argument('--batch_norm', type=int, default=0, help='use batch normalization over input embeddings (1=yes)')
    parser.add_argument('--seq_length', type=int, default=35, help='number of timesteps to unroll for')
    parser.add_argument('--batch_size', type=int, default=32, help='number of sequences to train on in parallel')
    parser.add_argument('--max_epochs', type=int, default=24, help='number of full passes through the training data')
    parser.add_argument('--max_grad_norm', type=float, default=5, help='normalize gradients at')
    # inputs
    parser.add_argument('--n_words', type=int, default=10000, help='max number of bpes in model')
    parser.add_argument('--n_bpes', type=int, default=5000, help='max number of bpes in model')
    parser.add_argument('--ngram', type=int, default=3, help='letter ngram')
    # bookkeeping
    parser.add_argument('--seed', type=int, default=3435, help='manual random number generator seed')
    parser.add_argument('--print_every', type=int, default=500, help='how many steps/minibatches between printing out the loss')
    parser.add_argument('--save_every', type=int, default=5, help='save every n epochs')
    parser.add_argument('--model_dir', type=str, default='cv', help='output directory where models get written')
    parser.add_argument('--savefile', type=str, default='char', help='filename to autosave the model to. Will be inside model_dir/')
    parser.add_argument('--EOS', type=str, default='+', help='<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')
    parser.add_argument('--skip_train', default=False, help='skip training', action='store_true')

    # parse input params
    params = parser.parse_args()
    np.random.seed(params.seed)

    assert params.predict_bpes == 1 or params.predict_bpes == 0, '-predict_bpes has to be 0 or 1'
    assert params.predict_words == 1 or params.predict_words == 0, '-predict_words has to be 0 or 1'
    assert (params.predict_words + params.predict_bpes) > 0, 'has to predict at least one of words or bpes'

    # global constants for certain tokens
    params.tokens = Tokens(
        EOS=params.EOS,
        UNK='|',    # unk word token
        START='{',  # start-of-word token
        END='}'    # end-of-word token
    )

    print('parsed parameters:')
    print(json.dumps(vars(params), indent = 2))

    main(params)
