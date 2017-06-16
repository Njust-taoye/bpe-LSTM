from __future__ import print_function

import codecs
import numpy as np
from os import path
import gc
import re, collections
from collections import Counter, OrderedDict, namedtuple
from itertools import izip

encoding='utf8'
# encoding='iso-8859-1'

Tokens = namedtuple('Tokens', ['EOS', 'UNK', 'START', 'END'])

def vocab_unpack(vocab):
    return vocab['idx2bpe'], vocab['bpe2idx'], vocab['idx2word'], vocab['word2idx']

class BatchLoaderWord:
    def __init__(self, tokens, data_dir, batch_size, seq_length, n_bpes, n_words, ngram):
        self.n_words = n_words
        self.n_bpes = n_bpes
        self.ngram = ngram
        train_file = path.join(data_dir, 'train' + str(n_bpes) + 'BPE.txt')
        valid_file = path.join(data_dir, 'valid' + str(n_bpes) + 'BPE.txt')
        test_file = path.join(data_dir, 'test' + str(n_bpes) + 'BPE.txt')
        input_files = [train_file, valid_file, test_file]
        train_txt = path.join(data_dir, 'train.txt')
        valid_txt = path.join(data_dir, 'valid.txt')
        test_txt = path.join(data_dir, 'test.txt')
        input_txt = [train_txt, valid_txt, test_txt]
        vocab_file = path.join(data_dir, 'vocab' + str(n_bpes) + '.npz')
        word_file = path.join(data_dir, 'data_word')
        bpe_file = path.join(data_dir, 'data_bpe')
        
        # construct a tensor with all the data
        if not (path.exists(vocab_file) or path.exists(word_file) or path.exists(bpe_file)):
            print('one-time setup: preprocessing input train/valid/test files in dir: ', data_dir)
            self.text_to_tensor(tokens, data_dir, input_files, input_txt, vocab_file, word_file, bpe_file)

        print('loading data files...')
        all_data = []
        all_data_bpe = []
        splits = 3 if data_dir == "data/ptb" else 2
        for split in range(splits):
            all_data.append(np.load("{}_{}words_{}.npy".format(word_file, self.n_words, split)))  # train, valid, test tensors
            all_data_bpe.append(np.load("{}_{}bpes_{}.npy".format(bpe_file, self.n_bpes, split)))
        vocab_mapping = np.load(vocab_file)
        self.idx2bpe, self.bpe2idx, self.idx2word, self.word2idx= vocab_unpack(vocab_mapping)
        self.vocab_size = len(self.idx2word)
        print('Word vocab size: %d, BPE vocab size: %d' % (len(self.idx2word), len(self.idx2bpe)))
        self.n_bpes = all_data_bpe[0].shape[1]
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
            data_bpe = all_data_bpe[split][:len(data)]
            if split < 2:
                rdata = data.reshape((batch_size, -1))
                rydata = ydata.reshape((batch_size, -1))
                rdata_bpe = data_bpe.reshape((batch_size, -1, self.n_bpes))
            else: # for test we repeat dimensions to batch size (easier but inefficient evaluation)
                nseq = (data_len + (seq_length - 1)) // seq_length
                rdata = data.copy()
                rdata.resize((1, nseq*seq_length))
                rdata = np.tile(rdata, (batch_size, 1))
                rydata = ydata.copy()
                rydata.resize((1, nseq*seq_length))
                rydata = np.tile(rydata, (batch_size, 1))
                rdata_bpe = data_bpe.copy()
                rdata_bpe.resize((1, nseq*seq_length, rdata_bpe.shape[1]))
                rdata_bpe = np.tile(rdata_bpe, (batch_size, 1, 1))
            # split in batches
            x_batches = np.split(rdata, rdata.shape[1]/seq_length, axis=1)
            y_batches = np.split(rydata, rydata.shape[1]/seq_length, axis=1)
            x_bpe_batches = np.split(rdata_bpe, rdata_bpe.shape[1]/seq_length, axis=1)
            print(len(x_batches), len(y_batches), len(x_bpe_batches))
            nbatches = len(x_batches)
            self.split_sizes.append(nbatches)
            assert len(x_batches) == len(y_batches)
            assert len(x_batches) == len(x_bpe_batches)
            self.all_batches.append((x_batches, y_batches, x_bpe_batches))

        self.word_vocab_size = len(self.idx2word)
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
            word = self.all_batches[split_idx][0][idx]
            sparse_ydata = self.all_batches[split_idx][1][idx]
            bpes = self.all_batches[split_idx][2][idx]
            # expand dims for sparse_cross_entropy optimization
            ydata = np.expand_dims(sparse_ydata, axis=2)
                    
            yield ({'word':word, 'bpes':bpes}, ydata)

    def sub_batch(self, split_idx):
        while True:
            self.batch_idx[split_idx] += 1
            # choose 1/10 train data to report loss
            if self.batch_idx[split_idx] >= self.split_sizes[split_idx] / 10:
                self.batch_idx[split_idx] = 0 # cycle around to beginning
            # pull out the correct next batch
            idx = self.batch_idx[split_idx]
            word = self.all_batches[split_idx][0][idx]
            sparse_ydata = self.all_batches[split_idx][1][idx]
            bpes = self.all_batches[split_idx][2][idx]
            # expand dims for sparse_cross_entropy optmization
            ydata = np.expand_dims(sparse_ydata, axis=2)
            
            yield ({'word':word, 'bpes':bpes}, ydata)

    ''' 
    # this function read in data files and generate the vocab in which letters
    # in words are separated by space
    def get_dict(self, input_files):
        print('Convert data files into vocab')
        wordcount = collections.defaultdict(int)
        for split in range(2):
          with open(input_files[split], 'r') as f:
            for line in f:
              words = line.split()
              for word in words:
                wordcount[word] += 1
        # split word into letters and add end of word </w> symbol
        word_dict = collections.defaultdict(int)
        for word, freq in wordcount.items():
          word_s = ""
          for i in range(len(word)):
            word_s += word[i] + ' '
          word_s += '</w>'
          word_dict[word_s] = freq
        return word_dict
    
    def get_stats(self, vocab):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out
    
    # convert bpe dictionary into a bpe vocab
    def get_bpe_vocab(self, bpe_dict, bpe_vocab):
       
    # convert input data files into tensors
    def text_to_bpeTensor(self, input_files, out_bpefile, n_bpe):
    '''
      
    def text_to_tensor(self, tokens, data_dir, input_files, input_txt, out_vocabfile, out_wordfile, out_bpefile):
        print('Processing text into tensors...')
        idx2word = [tokens.UNK]
        word2idx = OrderedDict()
        word2idx[tokens.UNK] = 0
        idx2bpe = [tokens.UNK]
        bpe2idx = OrderedDict()
        bpe2idx[tokens.UNK] = 0
        split_counts = []

        prog = re.compile('\s+')
        bpecount = Counter()
        wordcount = Counter()
        splits = 3 if data_dir == "data/ptb" else 2
        for split in range(splits): # split = 0 (train), 1 (val), or 2 (test)

            def update(bpe, check):
                if check == 0: # update bpe if check is zero
                    bpecount.update([bpe])
                else: # update word if check is not zero
                    wordcount.update([bpe])
            
            # compute word counts in train/valid/test 
            f_txt = codecs.open(input_txt[split], 'r', encoding)
            counts = 0
            for line in f_txt:
                # replace '<unk>' in PTB with the UNK symbol
                if data_dir == "data/ptb":
                    line = line.replace('<unk>', tokens.UNK)
                words = prog.split(line)
                for word in filter(None, words):
                    update(word, 1)
                    counts += 1
                if tokens.EOS != '':
                    update(tokens.EOS, 1)
                    counts += 1 
            f_txt.close()
            split_counts.append(counts)
                
            # compute bpe counts in train/valid/test 
            f = codecs.open(input_files[split], 'r', encoding)
            for line in f:
                # replace '<unk>' in PTB with the UNK symbol
                if data_dir == "data/ptb":
                    line = line.replace('<unk>', tokens.UNK)
                bpes = prog.split(line)
                for bpe in filter(None, bpes):
                    update(bpe, 0)
                if tokens.EOS != '':
                    update(tokens.EOS, 0)
            f.close()
                    
        print('Most frequent bpes:', len(bpecount))
        for ii, ww in enumerate(bpecount.most_common(self.n_bpes - 1)):
            bpe = ww[0]
            bpe2idx[bpe] = ii + 1
            idx2bpe.append(bpe)
            if ii < 3: print(bpe)

        print('Most frequent words:', len(wordcount))
        for ii, ww in enumerate(wordcount.most_common(self.n_words - 1)):
            word = ww[0]
            word2idx[word] = ii + 1
            idx2word.append(word)
            if ii < 3: print(word) 
        print('Word2idx size: %d' % len(word2idx))
        print('Bpe2idx size: %d' % len(bpe2idx))
        if splits == 3:
            print('Token count: train %d, val %d, test %d' % (split_counts[0], split_counts[1], split_counts[2]))
        else:
            print('Token count: train %d, val %d' % (split_counts[0], split_counts[1]))
            
        self.n_bpes = min(len(bpecount), self.n_bpes)
        self.n_words = min(len(wordcount), self.n_words)
        
        counts = []
        for split in range(splits):  # split = 0 (train), 1 (val), or 2 (test)
            # Preallocate the tensors we will need.
            # Watch out the second one needs a lot of RAM.
            output_words = np.empty(split_counts[split], dtype='int32')
            output_bpes = np.zeros((split_counts[split], self.n_bpes), dtype='float32')

            f = codecs.open(input_files[split], 'r', encoding)
            f_txt = codecs.open(input_txt[split], 'r', encoding)
            word_num = 0
            for line_bpe, line_word in izip(f, f_txt):
                # replace '<unk>' in PTB with the UNK symbol
                if data_dir == "data/ptb":
                    line_bpe = line.replace('<unk>', tokens.UNK)
                    line_word = line.replace('<unk>', tokens.UNK)
                bpes = prog.split(line_bpe)
                words = prog.split(line_word)
                i_bpe = 0
                for word in filter(None, words):
                    # put word in tensor
                    output_words[word_num] = word2idx[word] if word in word2idx else word2idx[tokens.UNK]
                    # put bpes of the word in bpe tensor
                    bpes_for_word = [0.0] * self.n_bpes
                    word_tmp = ""
                    while i_bpe < len(bpes):
                      word_tmp += str(bpes[i_bpe])
                      if bpes[i_bpe] in bpe2idx:
                        bpes_for_word[bpe2idx[bpes[i_bpe]]] += 1
                      else:
                        bpes_for_word[bpe2idx[tokens.UNK]] += 1
                      i_bpe += 1
                      # find word boundary to reconstruct the word
                      # word will ends with "</w>" or nothing
                      if word_tmp[:-4] == word or (len(word_tmp) <= 4  and word_tmp == word):
                        '''
                        if word_num < 50:
                          print("bpe: " + bpes[i_bpe - 1] + " word: " + word + " word_tmp: " + word_tmp)
                        '''
                        output_bpes[word_num, : self.n_bpes] = bpes_for_word
                        word_tmp = ""
                        word_num += 1
                        bpes_for_word = [0.0] * self.n_bpes
                        break
                '''
                if word_num < 50:
                    print(i_bpe)
                    print(len(bpes))
                    print("num words in first line: ", word_num)
                '''
                if tokens.EOS != '':   # PTB does not have <eos> so we add a character for <eos> tokens
                    output_words[word_num] = word2idx[tokens.EOS] if tokens.EOS in word2idx else word2idx[tokens.UNK]
                    bpes_for_word = [0.0] * self.n_bpes
                    if tokens.EOS in bpe2idx:
                        bpes_for_word[bpe2idx[tokens.EOS]] += 1
                    else:
                        bpes_for_word[bpe2idx[tokens.UNK]] += 1
                    output_bpes[word_num, : self.n_bpes] = bpes_for_word
                    word_num += 1
                 
            f.close()
            counts.append(word_num)
            print("split:{0}, counts: {1}".format(split, word_num))
            wordfile_split = "{}_{}words_{}.npy".format(out_wordfile, self.n_words, split)
            print('saving ', wordfile_split)
            np.save(wordfile_split, output_words)
            bpes_split = "{}_{}bpes_{}".format(out_bpefile, self.n_bpes, split)
            print('saving ', bpes_split)
            np.save(bpes_split, output_bpes)
        
        if splits == 3:
            print('Token count: train %d, val %d, test %d' % (counts[0], counts[1], counts[2]))
        else:
            print('Token count: train %d, val %d' % (counts[0], counts[1]))

        # save output preprocessed files
        print('saving ', out_vocabfile)
        np.savez(out_vocabfile, idx2bpe=idx2bpe, bpe2idx=bpe2idx, idx2word=idx2word, word2idx=word2idx)
