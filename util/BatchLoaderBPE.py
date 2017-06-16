from __future__ import print_function

# Modified from https://github.com/karpathy/char-rnn
# This version is for cases where one has already segmented train/val/test splits
import codecs
import numpy as np
from os import path
import gc
import re, collections
from collections import Counter, OrderedDict, namedtuple

encoding='utf8'
# encoding='iso-8859-1'

Tokens = namedtuple('Tokens', ['EOS', 'UNK', 'START', 'END'])

def vocab_unpack(vocab):
    return vocab['idx2bpe'], vocab['bpe2idx']

class BatchLoaderBPE:
    def __init__(self, tokens, data_dir, batch_size, seq_length, n_bpes, ngram):
        self.n_bpes = n_bpes
        self.ngram = ngram
        train_file = path.join(data_dir, 'train' + str(n_bpes) + 'BPE.txt')
        valid_file = path.join(data_dir, 'valid' + str(n_bpes) + 'BPE.txt')
        test_file = path.join(data_dir, 'test' + str(n_bpes) + 'BPE.txt')
        input_files = [train_file, valid_file, test_file]
        vocab_file = path.join(data_dir, 'vocab_predict_' + str(n_bpes) + '.npz')
        bpe_file = path.join(data_dir, 'data_predict_bpe')
        
        # construct a tensor with all the data
        if not (path.exists(vocab_file) or path.exists(bpe_file)):
            print('one-time setup: preprocessing input train/valid/test files in dir: ', data_dir)
            self.text_to_tensor(tokens, data_dir, input_files, vocab_file, bpe_file)

        print('loading data files...')
        all_data = []
        splits = 3 if data_dir == "data/ptb" else 2
        for split in range(splits):
            all_data.append(np.load("{}_{}bpes_{}.npy".format(bpe_file, self.n_bpes, split)))  # train, valid, test tensors
        vocab_mapping = np.load(vocab_file)
        self.idx2bpe, self.bpe2idx = vocab_unpack(vocab_mapping)
        self.vocab_size = len(self.idx2bpe)
        print('Bpe vocab size: %d' % len(self.idx2bpe))
        # create word-char mappings
        # cut off the end for train/valid sets so that it divides evenly
        # test set is not cut off
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.data_sizes = []
        self.split_sizes = []
        self.all_batches = []
        print('reshaping tensors...')
        for split, data in enumerate(all_data):
            data_len = data.shape[0]
            self.data_sizes.append(data_len)
            if split < 2 and data_len % (batch_size * seq_length) != 0:
                data = data[:batch_size * seq_length * (data_len // (batch_size * seq_length))]
            ydata = data.copy()
            ydata[0:-1] = data[1:]
            ydata[-1] = data[0]
            if split < 2:
                rdata = data.reshape((batch_size, -1))
                rydata = ydata.reshape((batch_size, -1))
            else: # for test we repeat dimensions to batch size (easier but inefficient evaluation)
                nseq = (data_len + (seq_length - 1)) // seq_length
                rdata = data.copy()
                rdata.resize((1, nseq*seq_length))
                rdata = np.tile(rdata, (batch_size, 1))
                rydata = ydata.copy()
                rydata.resize((1, nseq*seq_length))
                rydata = np.tile(rydata, (batch_size, 1))
            # split in batches
            x_batches = np.split(rdata, rdata.shape[1]/seq_length, axis=1)
            y_batches = np.split(rydata, rydata.shape[1]/seq_length, axis=1)
            print(len(x_batches), len(y_batches))
            nbatches = len(x_batches)
            self.split_sizes.append(nbatches)
            assert len(x_batches) == len(y_batches)
            self.all_batches.append((x_batches, y_batches))

        self.bpe_vocab_size = len(self.idx2bpe)
        if data_dir == "data/pbt":
            self.batch_idx = [0,0,0]
            print('data load done. Number of batches in train: %d, val: %d, test: %d' \
              % (self.split_sizes[0], self.split_sizes[1], self.split_sizes[2]))
        else:
            self.batch_idx = [0,0]
            print('data load done. Number of batches in train: %d, val: %d' \
              % (self.split_sizes[0], self.split_sizes[1]))
        gc.collect()

    def reset_batch_pointer(self, split_idx, batch_idx=0):
        self.batch_idx[split_idx] = batch_idx

    def next_batch(self, split_idx):
        while True:
            # split_idx is integer: 0 = train, 1 = val, 2 = test
            self.batch_idx[split_idx] += 1
            if self.batch_idx[split_idx] >= self.split_sizes[split_idx]:
                self.batch_idx[split_idx] = 0 # cycle around to beginning

            # pull out the correct next batch
            idx = self.batch_idx[split_idx]
            bpe = self.all_batches[split_idx][0][idx]
            sparse_ydata = self.all_batches[split_idx][1][idx]
            # expand dims for sparse_cross_entropy optimization
            ydata = np.expand_dims(sparse_ydata, axis=2)
                    
            yield ({'bpe':bpe}, ydata)

    def sub_batch(self, split_idx):
        while True:
            self.batch_idx[split_idx] += 1
            # choose 1/10 train data to report loss
            if self.batch_idx[split_idx] >= self.split_sizes[split_idx] / 10:
                self.batch_idx[split_idx] = 0 # cycle around to beginning
            # pull out the correct next batch
            idx = self.batch_idx[split_idx]
            bpe = self.all_batches[split_idx][0][idx]
            sparse_ydata = self.all_batches[split_idx][1][idx]
            # expand dims for sparse_cross_entropy optmization
            ydata = np.expand_dims(sparse_ydata, axis=2)
            
            yield ({'bpe':bpe}, ydata)
      
    def text_to_tensor(self, tokens, data_dir, input_files, out_vocabfile, out_tensorfile):
        print('Processing text into tensors...')
        idx2bpe = [tokens.UNK]
        bpe2idx = OrderedDict()
        bpe2idx[tokens.UNK] = 0
        split_counts = []

        prog = re.compile('\s+')
        bpecount = Counter()
        splits = 3 if data_dir == "data/ptb" else 2
        for split in range(splits): # split = 0 (train), 1 (val), or 2 (test)

            def update(bpe):
                bpecount.update([bpe])
            
            f = codecs.open(input_files[split], 'r', encoding)
            counts = 0
            for line in f:
                # replace '<unk>' in PTB with the UNK symbol
                if data_dir == "data/ptb":
                    line = line.replace('<unk>', tokens.UNK)
                bpes = prog.split(line)
                for bpe in filter(None, bpes):
                    update(bpe)
                    counts += 1
                if tokens.EOS != '':
                    update(tokens.EOS)
                    counts += 1 # PTB uses \n for <eos>, so need to add one more token at the end
            f.close()
            split_counts.append(counts)
                    
        print('Most frequent bpes:', len(bpecount))
        for ii, ww in enumerate(bpecount.most_common(self.n_bpes - 1)):
            bpe = ww[0]
            bpe2idx[bpe] = ii + 1
            idx2bpe.append(bpe)
            if ii < 3: print(bpe)
        print('Bpe2idx size: %d' % len(bpe2idx))
        if splits == 3:
            print('Token count: train %d, val %d, test %d' % (split_counts[0], split_counts[1], split_counts[2]))
        else:
            print('Token count: train %d, val %d' % (split_counts[0], split_counts[1]))
        
        for split in range(splits):  # split = 0 (train), 1 (val), or 2 (test)
            # Preallocate the tensors we will need.
            output_tensor = np.empty(split_counts[split], dtype='int32')

            def append(bpe, bpe_num):
                output_tensor[bpe_num] = bpe2idx[bpe] if bpe in bpe2idx else bpe2idx[tokens.UNK]
                return bpe_num + 1

            f = codecs.open(input_files[split], 'r', encoding)
            bpe_num = 0
            for line in f:
                # replace '<unk>' in PTB with the UNK symbol
                if data_dir == "data/ptb":
                    line = line.replace('<unk>', tokens.UNK)
                bpes = prog.split(line)
                for bpe in filter(None, bpes):
                    bpe_num = append(bpe, bpe_num)
                if tokens.EOS != '':   # PTB does not have <eos> so we add a character for <eos> tokens
                    bpe_num = append(tokens.EOS, bpe_num)   # other datasets don't need this
            f.close()
            tensorfile_split = "{}_{}bpes_{}.npy".format(out_tensorfile, self.n_bpes, split)
            print('saving ', tensorfile_split)
            np.save(tensorfile_split, output_tensor)

        # save output preprocessed files
        print('saving ', out_vocabfile)
        np.savez(out_vocabfile, idx2bpe=idx2bpe, bpe2idx=bpe2idx)
