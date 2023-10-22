import os
from math import floor,ceil
from random import shuffle,seed
from collections import Counter

def read_dataset(filename,bos='<bos>',eos='<eos>'):
    """
    Generator for sentences in a wikitext style corpus
    :param filename: a string
    :param bos: the begin of sentence token
    :param eos: the end of sentence token
    :yields: a list of tokens (strings) one at a time
    """
    istream = open(filename)
    for line in istream:
        line = line.split('=')[0]
        if line and not line.isspace():
            tokens = [bos] + line.split() + [eos]
            yield tokens
    istream.close()

def vocabulary(corpus, max_size=-1, unk=None,pad=None):
    """
    Generates the encoding string to int and vice-versa
    for a whole corpus.

    The encoding is frequency sensitive: the indexing is a frequency rank
    where most frequent tokens have lowest indices (except for special tokens)

    :param corpus:an iterable of sentences. A sentence is a list of strings
    :param max_size : int max number of elements in the vocab
    :return: a couple. a dict mapping strings to int and a list mapping int to strings
    """
    vocab = Counter()
    for sentence in corpus:
         vocab.update(sentence)

    idx2str = [pad]
    if unk and unk not in vocab:
        idx2str.append(unk)
    idx2str.extend(tok for tok,count in vocab.most_common(max_size))
    str2idx = {token:idx for (idx,token) in enumerate(idx2str)}
    print('Vocabulary size = %d'%(len(idx2str)))
    return (str2idx,idx2str)

def pad(sentence,pad_size,pad_token):

    return sentence + [pad_token] * (pad_size-len(sentence))


class Dataset:

        def __init__(self,filename,bos='<bos>',eos='<eos>',unk='<unk>',parentencoding=None,max_vocab_size=-1):

            self.sentences = []
            # reads sentences and destructively performs truncation (attempts to avoid memory explosion)
            if filename:
                self.sentences = list(read_dataset(filename, bos, eos))

            if type(parentencoding) == str:
                istream = open(os.path.join(parentencoding, 'tokcodes'))
                unk = istream.readline().strip()
                parentencoding = [line.strip() for line in istream]
                istream.close()
            if type(parentencoding) == list:
                self.pad_token = parentencoding[0]
                self.unk_token = unk
                self.idx2tok = parentencoding
                self.tok2idx = {token:idx for idx,token in enumerate(self.idx2tok)}
            else:
                self.pad_token = '<pad>'
                self.unk_token = unk
                self.tok2idx, self.idx2tok = vocabulary(self.sentences,pad=self.pad_token,unk=self.unk_token,max_size=max_vocab_size)
            self.pad_idx = self.tok2idx[self.pad_token]
            self.unk_idx = self.tok2idx[self.unk_token]

        @property
        def encoding(self):
            return self.idx2tok

        def save(self,dirname):
            """
            Saves the dataset *encoding* to file
            :param dirname:
            :return:
            """
            ostream = open(os.path.join(dirname,'tokcodes'),'w')
            print(self.unk_token,file=ostream)
            ostream.write('\n'.join(self.encoding))
            ostream.close()

        def vocab_size(self):
            return len(self.idx2tok)

        def num_batches(self,batch_size,bptt_len=10000,world_size=1):
            """
            An ~expected number of 'physical' batches, this includes duplications of long batches for trunctated backprop.
            A single gpu will process exactly this number of batches during every epoch.

            Args:
                batch_size(int): size of a batch in sentences
            Returns: an int
            """
            B = 0
            N = len(self.sentences)
            idxes = list(range(N))
            idxes.sort(key=lambda x: len(self.sentences[x]))
            for sidx in range(0,N,batch_size):
                eidx = min(sidx + batch_size, N)
                batchlen = max([len(self.sentences[idx]) - 1 for idx in idxes[sidx:eidx]])
                B += ceil(batchlen/bptt_len)
            return floor(B/world_size)


        def init_epoch(self,init_seed,batch_size,keep_order,worker_id,world_size):
            """
            Performs the shuffling and distribution of a dataset in multiprocessing context.
            Args:
                init_seed  (int): a seed to init the random generator with
                batch_size (int): the size of a full batch
            :return:
            """
            seed(init_seed)
            if type(worker_id) == str:
                worker_id =  int(worker_id[-1])
            N = len(self.sentences)
            self.idxes  = list(range(N))
            self.sidxes = list(range(0 + worker_id, N, batch_size * world_size))  # batch start idxes
            if not keep_order:
                shuffle(self.idxes)
                self.idxes.sort(key=lambda x: len(self.sentences[x]))
                shuffle(self.sidxes)


        def generate_batch(self,batch_size,worker_id = 0, world_size = 1,init_seed=0,bptt_len=10000,keep_order=False):
            """
            Generates a batch of data.
            The generator ensures that every GPU receives the exact same number of batches
            This conservative method prevents deadlocks in multiprocessing.

            Args:
                batch_size (int):size of generated batches
                init_seed  (int):random number seed
                keep_order (int):shuffles the data (or not)
                bptt_len   (int):max number of tokens in a given batch chunk
            Returns a subset of the data as a triple (X,Y,first)
            where first is true if the batch is the first for a set of sentences and false otherwise
            """
            self.init_epoch(init_seed,batch_size,keep_order,worker_id,world_size)
            nbatches = self.num_batches(batch_size, world_size=world_size, bptt_len=bptt_len)
            sidxes = iter(self.sidxes)
            B    = 0
            N    = len(self.sentences)
            while True:
               try:
                   sidx = next(sidxes)
               except StopIteration:
                   sidxes = iter(self.sidxes)
                   sidx   = next(sidxes)

               eidx = min(sidx+batch_size,N)
               batchlen = max([len(self.sentences[idx])-1 for idx in self.idxes[sidx:eidx]])
               X = [self.sentences[idx][:-1] for idx in self.idxes[sidx:eidx] ]
               Y = [self.sentences[idx][1:] for idx in self.idxes[sidx:eidx] ]
               #padding
               X = [pad(x,batchlen,pad_token = self.pad_token) for x in X]
               Y = [pad(y,batchlen,pad_token = self.pad_token) for y in Y]
               #coding
               xinput  = [ [self.tok2idx.get(token,self.tok2idx[self.unk_token]) for token in x] for x in X]
               youtput = [ [self.tok2idx.get(token,self.tok2idx[self.unk_token]) for token in y] for y in Y]
               #transpose matrices (batch,seq) -> (seq,batch)
               xinput  = list( zip(*xinput))
               youtput = list(zip(*youtput))

               #truncates long batches and returns
               first = True
               for idx in range(0, batchlen, bptt_len):
                  xchunk = xinput[idx:idx + bptt_len]
                  ychunk = youtput[idx:idx + bptt_len]
                  yield (xchunk,ychunk, first)
                  B += 1
                  first = False
                  if B >= nbatches: # <= exit loop here
                      return


#Example usage
if __name__ == '__main__':

    trainset = Dataset('wiki.train.tokens')
    validset = Dataset('wiki.valid.tokens',unk=trainset.unk_token, parentencoding=trainset.encoding)
    testset  = Dataset('wiki.test.tokens',unk=trainset.unk_token,  parentencoding=trainset.encoding)
    for (X,Y) in trainset.generate_batch(32):
        print(X,Y)

