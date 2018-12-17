# intro
Seq2seq module is chosen. 

Popular Recurrent Neural Networks modules include **RNN LSTM GRU**

From here **LSTM** is chosen

we train samples from a lot of QA pairs. we can download opensource dataset from Twitter,Reddit

# data preprocess
UNK: unknown words or words more than given length
Go: decoder start signal
EOS: answer end signal
PAD: implement shor words(ensure same length QA pair in Seq2seq's intput and output)

```python
limit = {
    'maxq': 10,
    'minq': 0,
    'maxa': 8,
    'mina': 3
}

UNK = 'unk'
GO = '<go>'
EOS = '<eos>'
PAD = '<pad>'
VOCAB_SIZE = 1000

```

filter from QA length limit

```python
def filter_data(sequences):
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences) // 2

    for i in range(0, len(sequences), 2):
        qlen, alen = len(sequences[i].split(' ')), len(sequences[i + 1].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(sequences[i])
                filtered_a.append(sequences[i + 1])
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len) * 100 / raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a
```

get words freqencies and topn words, also the index
```python
def index_(tokenized_sentences, vocab_size):
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = freq_dist.most_common(vocab_size)
    index2word = [GO] + [EOS] + [UNK] + [PAD] + [x[0] for x in vocab]
    word2index = dict([(w, i) for i, w in enumerate(index2word)])
    return index2word, word2index, freq_dist

```

pad part
eg: **"how are you"** if size fixed 10 -> **"how are you" + "pad * 6"**
for decoder **"fine thank you"-> "** go fine thank you eos pad pad pad pad pad "
for target same as decoder, but a positon shift **"fine thank you eos pad pad pad pad pad pad"**
```python
def zero_pad(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)
    # +2 dues to '<go>' and '<eos>'
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa'] + 2], dtype=np.int32)
    idx_o = np.zeros([data_len, limit['maxa'] + 2], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'], 1)
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'], 2)
        o_indices = pad_seq(atokenized[i], w2idx, limit['maxa'], 3)
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)
        idx_o[i] = np.array(o_indices)

    return idx_q, idx_a, idx_o


def pad_seq(seq, lookup, maxlen, flag):
    if flag == 1:
        indices = []
    elif flag == 2:
        indices = [lookup[GO]]
    elif flag == 3:
        indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    if flag == 1:
        return indices + [lookup[PAD]] * (maxlen - len(seq))
    elif flag == 2:
        return indices + [lookup[EOS]] + [lookup[PAD]] * (maxlen - len(seq))
    elif flag == 3:
        return indices + [lookup[EOS]] + [lookup[PAD]] * (maxlen - len(seq) + 1)

```








# train process
1. run 'data.py' to produce some files we needed.
2. run 'train.py' to train the model.
3. run 'test_model.py' to predict.

# implementation
> can be used as 
