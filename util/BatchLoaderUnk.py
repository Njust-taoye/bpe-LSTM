
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

Tokens = namedtuple('Tokens', ['EOS', 'UNK', 'START', 'END', 'ZEROPAD'])

def vocab_unpack(vocab):
    return vocab['idx2bpe'], vocab['bpe2idx']

class BatchLoaderUnk:
    def __init__(self, tokens, data_dir, batch_size, seq_length, n_bpes):
        self.n_bpes = n_bpes
        train_file = path.join(data_dir, 'train' + str(n_bpes) + 'BPE.noEow.txt')
        valid_file = path.join(data_dir, 'valid' + str(n_bpes) + 'BPE.noEow.txt')
        test_file = path.join(data_dir, 'test' + str(n_bpes) + 'BPE.noEow.txt')
        input_files = [train_file, valid_file, test_file]
        vocab_file = path.join(data_dir, 'vocab' + str(n_bpes) + '.noEow.npz')
        tensor_file = path.join(data_dir, 'data_bpe')
        
        # construct a tensor with all the data
        if not (path.exists(vocab_file) or path.exists(tensor_file)):
            print 'one-time setup: preprocessing input train/valid/test files in dir: ', data_dir
            self.text_to_tensor(tokens, input_files, vocab_file, tensor_file)

        print('loading data files...')
        all_data = []
        all_data_char = []
        for split in range(3):
            all_data.append(np.load("{}_{}_{}.npy".format(tensor_file, self.n_bpes, split)))  # train, valid, test tensors
        vocab_mapping = np.load(vocab_file)
        self.idx2bpe, self.bpe2idx = vocab_unpack(vocab_mapping)
        self.vocab_size = len(self.idx2bpe)
        print 'Bpe vocab size: %d' % len(self.idx2bpe)
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

        self.batch_idx = [0,0,0]
        self.bpe_vocab_size = len(self.idx2bpe)
        print 'data load done. Number of batches in train: %d, val: %d, test: %d' \
              % (self.split_sizes[0], self.split_sizes[1], self.split_sizes[2])
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
            # expand dims for sparse_cross_entropy optimization
            ydata = np.expand_dims(sparse_ydata, axis=2)
                    
            yield ({'bpe':word}, ydata)
    ''' 
    # this function read in data files and generate the vocab in which letters
    # in words are separated by space
    def get_dict(self, input_files):
        print 'Convert data files into vocab'
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
      
    def text_to_tensor(self, tokens, input_files, out_vocabfile, out_tensorfile):
        print 'Processing text into tensors...'
        idx2bpe = [tokens.UNK] # unknown word token
        bpe2idx = OrderedDict()
        bpe2idx[tokens.UNK] = 0
        split_counts = []

        # first go through train/valid/test to get max word length
        # if actual max word length is smaller than specified
        # we use that instead. this is inefficient, but only a one-off thing so should be fine
        # also counts the number of tokens
        prog = re.compile('\s+')
        wordcount = Counter()
        for split in range(3): # split = 0 (train), 1 (val), or 2 (test)

            def update(word):
                # print("word before update: ", word)
                if word[0] == tokens.UNK:
                    if len(word) > 1: # unk token with character info available
                        word = word[1:]
                else:
                    wordcount.update([word])
            
            f = codecs.open(input_files[split], 'r', encoding)
            counts = 0
            for line in f:
                line = line.replace('<unk>', tokens.UNK)  # replace unk with a single character
                line = line.replace(tokens.START, '')  # start-of-word token is reserved
                line = line.replace(tokens.END, '')  # end-of-word token is reserved
                words = prog.split(line)
                for word in filter(None, words):
                    update(word)
                    counts += 1
                if tokens.EOS != '':
                    update(tokens.EOS)
                    counts += 1 # PTB uses \n for <eos>, so need to add one more token at the end
            f.close()
            split_counts.append(counts)
                    
        print 'Most frequent bpes:', len(wordcount)
        for ii, ww in enumerate(wordcount.most_common(self.n_bpes - 1)):
            word = ww[0]
            bpe2idx[word] = ii + 1
            idx2bpe.append(word)
            if ii < 3: print word
        '''
        print 'Word counts:'
        for ii, cc in enumerate(wordcount.most_common()):
            print ii, cc[0].encode(encoding), cc[1]
        '''
        print 'Word2idx size: %d' % len(bpe2idx)
        print 'Token count: train %d, val %d, test %d' % (split_counts[0], split_counts[1], split_counts[2])
        
        for split in range(3):  # split = 0 (train), 1 (val), or 2 (test)
            # Preallocate the tensors we will need.
            # Watch out the second one needs a lot of RAM.
            output_tensor = np.empty(split_counts[split], dtype='int32')

            def append(word, word_num):
                # get word representation
                if word[0] == tokens.UNK and len(word) > 1: # unk token with character info available
                    word = word[1:]
                    output_tensor[word_num] = bpe2idx[tokens.UNK]
                else:
                    output_tensor[word_num] = bpe2idx[word] if word in bpe2idx else bpe2idx[tokens.UNK]
                return word_num + 1

            f = codecs.open(input_files[split], 'r', encoding)
            word_num = 0
            for line in f:
                line = line.replace('<unk>', tokens.UNK)  # replace unk with a single character
                line = line.replace(tokens.START, '')  # start-of-word token is reserved
                line = line.replace(tokens.END, '')  # end-of-word token is reserved
                words = prog.split(line)
                for rword in filter(None, words):
                    word_num = append(rword, word_num)
                if tokens.EOS != '':   # PTB does not have <eos> so we add a character for <eos> tokens
                    word_num = append(tokens.EOS, word_num)   # other datasets don't need this
            f.close()
            tensorfile_split = "{}_{}_{}.npy".format(out_tensorfile, self.n_bpes, split)
            print 'saving ', tensorfile_split
            np.save(tensorfile_split, output_tensor)

        # save output preprocessed files
        print 'saving ', out_vocabfile
        np.savez(out_vocabfile, idx2bpe=idx2bpe, bpe2idx=bpe2idx)
