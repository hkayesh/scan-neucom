import random
import numpy as np
import tensorflow as tf


# Configure a new global `tensorflow` session
from keras import backend as K

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# End of setting seed values
######################################

import sys
import os, shutil
import optparse
import approximateMatchCausal as approximateMatch
import prep
import nltk
import time
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from keras.models import model_from_json, Sequential
from keras.layers import Bidirectional, Convolution1D, BatchNormalization, Activation
from keras.layers import Embedding, LSTM, Dense, Reshape, TimeDistributed
from keras import callbacks
from keras.preprocessing import sequence
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.models import Model

from keras_multi_head import MultiHeadAttention

from word2vec_twitter_model import word2vecReader as w2vec
from settings import SETTINGS

os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = SETTINGS[
    'DYLD_FALLBACK_LIBRARY_PATH']  # Mitigates some OS X / Theano problems


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def vectorize_set(lexlists, maxlen, V):
    nb_samples = len(lexlists)
    X = np.zeros([nb_samples, maxlen, V])
    for i, lex in enumerate(lexlists):
        for j, tok in enumerate(lex):
            X[i, j, tok] = 1
    return X

def embed_set(lexlists, toklists, maxlen, w2vModel):
    def pad_toks(toklist, padlen):
        padded = ['PAD'] * (padlen - len(toklist))
        padded += toklist
        return np.array(padded)

    ## TODO: Normalize URLs, digits, etc

    dim = w2vModel.syn0norm.shape[1]
    nb_samples = len(lexlists)
    X = np.zeros([nb_samples, maxlen, dim])
    for i, (lex, toklist) in enumerate(zip(lexlists, toklists)):
        toklist = pad_toks(toklist, maxlen)
        for j, tok in enumerate(toklist):
            if tok != 'PAD':
                idx = w2vModel.vocab.get(tok, w2vModel.vocab['the']).index
                vec = w2vModel.syn0norm[idx]
                X[i, j] = vec
    return X


def learning_curve(history, pltname='history.pdf', preddir=None, fileprefix=''):
    '''
    Update: changed by Humayun to make chart look better

    Plot Validation accuracy/loss and optionally Approximate Match F1
    for each epoch
    :param history: keras.callbacks.History object
    :param preddir: directory containing approximate match results for ApproximateMatch callback
    :return:
    '''
    num_epochs = len(history.history['val_loss'])
    n = range(num_epochs)

    approxmatch = []
    for i in n:
        f1 = \
        open(os.path.join(preddir, fileprefix + 'approxmatch_epoch' + str(i)), 'rU').readlines()[-1].strip().split()[-1]
        approxmatch.append(float(f1))

    plt.clf()
    plt.plot(n, history.history['loss'], '-*', linewidth=1, alpha=0.9, label='Trn Loss')
    # plt.plot(n, history.history['acc'], '-D', linewidth=1, alpha=0.9, label='Trn Acc')
    plt.plot(n, history.history['val_loss'], '->', linewidth=1, alpha=0.9, label='Val Loss')
    # plt.plot(n, history.history['val_acc'], '-<', linewidth=1, alpha=0.9, label='Val Acc')
    plt.plot(n, approxmatch, '-D', linewidth=1, alpha=0.9, label='Val F1-score')

    plt.xlabel('Epochs')
    plt.ylabel('Scores')
    plt.legend()

    plt.savefig(pltname)
    plt.close()
    sys.stderr.write('Max approximate match F1: %0.2f\n' % max(approxmatch))
    return np.max(approxmatch), np.argmax(approxmatch)


def predict_score(model, x, toks, y, pred_dir, i2l, padlen, metafile=None, fileprefix=''):
    ## GRAPH (BIDIRECTIONAL)
    start_time = time.time()
    pred_probs = model.predict(x)
    prediction_time = time.time() - start_time
    test_loss = model.evaluate(x, y, batch_size=1, verbose=0)
    pred = np.argmax(pred_probs, axis=2)

    N = len(toks)

    # If the name of a metafile is passed, simply write this round of predictions to file
    if metafile:
        meta = open(metafile, 'a')

    fname = os.path.join(pred_dir, fileprefix + 'approxmatch_test')
    with open(fname, 'w') as fout:
        for i in range(N):
            bos = 'BOS\tO\tO\n'
            fout.write(bos)
            if metafile:
                meta.write(bos)

            sentlen = len(toks[i])
            startind = padlen - sentlen

            preds = [i2l[j] for j in pred[i][startind:]]
            actuals = [i2l[j] for j in np.argmax(y[i], axis=1)[startind:]]
            for (w, act, p) in zip(toks[i], actuals, preds):
                line = '\t'.join([w, act, p]) + '\n'
                fout.write(line)
                if metafile:
                    meta.write(line)

            eos = 'EOS\tO\tO\n'
            fout.write(eos)
            if metafile:
                meta.write(eos)
    scores = approximateMatch.get_approx_match(fname)
    scores['loss'] = test_loss
    scores['pred_time'] = prediction_time
    if metafile:
        meta.close()

    with open(fname, 'a') as fout:
        fout.write('\nTEST Approximate Matching Results:\n  ADR: Precision ' + str(scores['p']) + ' Recall ' + str(
            scores['r']) + ' F1 ' + str(scores['f1']))
    return scores

def get_causality_features(tokens_sets, y_sets, vectorizer, maxlen_seq):

    prefix_features = []
    midfix_features = []
    postfix_features = []

    feature_dim = len(vectorizer.get_feature_names())

    for tokens_set, y_set in zip(tokens_sets, y_sets):
        med_index = tokens_set.index('<medicine>')  # index of the medicine word in the tweet

        prefix_vectors = []
        midfix_vectors = []
        postfix_vectors = []
        index = 0
        for tok, y in zip(tokens_set, y_set):

            if index < med_index:
                prefix = tokens_set[:index]
                midfix = tokens_set[index + 1:med_index]
                postfix = tokens_set[med_index + 1:] if med_index < len(
                    tokens_set) - 1 else []  # if med_index is the last index, then the prefix is []
            elif index > med_index:
                prefix = tokens_set[:med_index]
                midfix = tokens_set[med_index + 1:index]
                postfix = tokens_set[index + 1:] if index < len(
                    tokens_set) - 1 else []  # if the current word is the last word in the list, the then prefix []
            else:
                prefix = []
                midfix = []
                postfix = []

            if len(prefix) > 0:
                prefix_vector = vectorizer.transform([' '.join(prefix)]).toarray()[0]
            else:
                prefix_vector = np.array([0] * feature_dim)

            if len(midfix) > 0:
                midfix_vector = vectorizer.transform([' '.join(midfix)]).toarray()[0]
            else:
                midfix_vector = np.array([0] * feature_dim)

            if len(postfix) > 0:
                postfix_vector = vectorizer.transform([' '.join(postfix)]).toarray()[0]
            else:
                postfix_vector = np.array([0] * feature_dim)

            prefix_vectors.append(prefix_vector)
            midfix_vectors.append(midfix_vector)
            postfix_vectors.append(postfix_vector)
            index += 1
        prefix_features.append(prefix_vectors)
        midfix_features.append(midfix_vectors)
        postfix_features.append(postfix_vectors)

    return prefix_features, midfix_features, postfix_features

def extract_causality_feature(dataset, idx2word, maxlen_seq):
    train_toks, valid_toks, test_toks, \
    train_lex, valid_lex, test_lex, \
    train_y, valid_y, test_y = dataset

    word2idx = dict(map(reversed, idx2word.items()))
    training_tweets = [' '.join(item_list) for item_list in train_toks]
    # vectorizer = TfidfVectorizer(ngram_range=(2, 3), max_features=300)
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), vocabulary=word2idx)
    vectorizer.fit(training_tweets)

    train_causal_features = np.array(get_causality_features(train_toks, train_y, vectorizer, maxlen_seq))
    valid_causal_features = np.array(get_causality_features(valid_toks, valid_y, vectorizer, maxlen_seq))
    test_causal_features = np.array(get_causality_features(test_toks, test_y, vectorizer, maxlen_seq))

    return train_causal_features, valid_causal_features, test_causal_features


def get_pos_tags(tok_lists):
    tagged_tok_lists = []
    for tok_list in tok_lists:
        tagged_tok_list = nltk.pos_tag(tok_list)
        assert len(tok_list) == len(tagged_tok_list)
        tags = [i[1] for i in tagged_tok_list]
        tagged_tok_lists.append(tags)

    return tagged_tok_lists

def model_scan(dataset, idx2word, idx2label, w2v, basedir, validate=True, num_epochs=30):
    train_toks, valid_toks, test_toks, \
    train_lex, valid_lex, test_lex, \
    train_y, valid_y, test_y = dataset

    maxlen_seq = max([len(l) for l in train_lex])
    if len(valid_lex) > 0:
        maxlen_seq = max(maxlen_seq, max([len(l) for l in valid_lex]))
    maxlen_seq = max(maxlen_seq, max([len(l) for l in test_lex]))

    nclasses = max(idx2label.keys()) + 1

    # Pad inputs to max sequence length and turn into one-hot vectors
    train_lex = sequence.pad_sequences(train_lex, maxlen=maxlen_seq)
    valid_lex = sequence.pad_sequences(valid_lex, maxlen=maxlen_seq)
    test_lex = sequence.pad_sequences(test_lex, maxlen=maxlen_seq)

    train_lex = embed_set(train_lex, train_toks, maxlen_seq, w2v)
    valid_lex = embed_set(valid_lex, valid_toks, maxlen_seq, w2v)
    test_lex = embed_set(test_lex, test_toks, maxlen_seq, w2v)

    train_y = sequence.pad_sequences(train_y, maxlen=maxlen_seq)
    valid_y = sequence.pad_sequences(valid_y, maxlen=maxlen_seq)
    test_y = sequence.pad_sequences(test_y, maxlen=maxlen_seq)

    train_y = vectorize_set(train_y, maxlen_seq, nclasses)
    valid_y = vectorize_set(valid_y, maxlen_seq, nclasses)
    test_y = vectorize_set(test_y, maxlen_seq, nclasses)

    # Prepare causality features
    train_causal_features, valid_causal_features, test_causal_features = extract_causality_feature(dataset, idx2word,
                                                                                                   maxlen_seq)
    train_prefix_causal_features = train_causal_features[0]
    train_midfix_causal_features = train_causal_features[1]
    train_postfix_causal_features = train_causal_features[2]

    valid_prefix_causal_features = valid_causal_features[0]
    valid_midfix_causal_features = valid_causal_features[1]
    valid_postfix_causal_features = valid_causal_features[2]

    test_prefix_causal_features = test_causal_features[0]
    test_midfix_causal_features = test_causal_features[1]
    test_postfix_causal_features = test_causal_features[2]

    train_prefix_causal_features = sequence.pad_sequences(train_prefix_causal_features, maxlen=maxlen_seq)
    train_midfix_causal_features = sequence.pad_sequences(train_midfix_causal_features, maxlen=maxlen_seq)
    train_postfix_causal_features = sequence.pad_sequences(train_postfix_causal_features, maxlen=maxlen_seq)

    valid_prefix_causal_features = sequence.pad_sequences(valid_prefix_causal_features, maxlen=maxlen_seq)
    valid_midfix_causal_features = sequence.pad_sequences(valid_midfix_causal_features, maxlen=maxlen_seq)
    valid_postfix_causal_features = sequence.pad_sequences(valid_postfix_causal_features, maxlen=maxlen_seq)

    test_prefix_causal_features = sequence.pad_sequences(test_prefix_causal_features, maxlen=maxlen_seq)
    test_midfix_causal_features = sequence.pad_sequences(test_midfix_causal_features, maxlen=maxlen_seq)
    test_postfix_causal_features = sequence.pad_sequences(test_postfix_causal_features, maxlen=maxlen_seq)

    # Prepare POS tag features
    train_pos_tags = get_pos_tags(train_toks)
    valid_pos_tags = get_pos_tags(valid_toks)
    test_pos_tags = get_pos_tags(test_toks)

    pos_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
    pos_tokenizer.fit_on_texts(train_pos_tags)

    train_pos_features = pos_tokenizer.texts_to_sequences(train_pos_tags)
    valid_pos_features = pos_tokenizer.texts_to_sequences(valid_pos_tags)
    test_pos_features = pos_tokenizer.texts_to_sequences(test_pos_tags)

    train_pos_features = sequence.pad_sequences(train_pos_features, maxlen=maxlen_seq)
    valid_pos_features = sequence.pad_sequences(valid_pos_features, maxlen=maxlen_seq)
    test_pos_features = sequence.pad_sequences(test_pos_features, maxlen=maxlen_seq)

    # train_xs = [train_lex, train_pos_features]
    # valid_xs = [valid_lex, valid_pos_features]
    # test_xs = [test_lex, test_pos_features]

    train_xs = [train_lex, train_prefix_causal_features, train_midfix_causal_features, train_postfix_causal_features, train_pos_features]
    valid_xs = [valid_lex, valid_prefix_causal_features, valid_midfix_causal_features, valid_postfix_causal_features, valid_pos_features]
    test_xs = [test_lex, test_prefix_causal_features, test_midfix_causal_features, test_postfix_causal_features, test_pos_features]

    ## Prepare input dimensions
    word_input = Input(shape=(maxlen_seq, train_lex.shape[2]))
    prefix_causal_input = Input(shape=(maxlen_seq, train_prefix_causal_features.shape[2]))
    midfix_causal_input = Input(shape=(maxlen_seq, train_midfix_causal_features.shape[2]))
    postfix_causal_input = Input(shape=(maxlen_seq, train_postfix_causal_features.shape[2]))
    pos_input = Input(shape=(maxlen_seq, ))

    word_features = Bidirectional(LSTM(80, return_sequences=True, activation='tanh',
                                       dropout=0.1, recurrent_dropout=0.1, name='forward'), merge_mode='ave')(word_input)

    prefix_causal_features = Convolution1D(filters=80, kernel_size=5, activation='relu', padding='same')(
        prefix_causal_input)
    prefix_causal_features = Dense(16, use_bias=False)(prefix_causal_features)
    prefix_causal_features = BatchNormalization()(prefix_causal_features)
    prefix_causal_features = Activation('relu')(prefix_causal_features)

    midfix_causal_features = Convolution1D(filters=80, kernel_size=5, activation='relu', padding='same')(
        midfix_causal_input)
    midfix_causal_features = Dense(32, use_bias=False)(midfix_causal_features)
    midfix_causal_features = BatchNormalization()(midfix_causal_features)
    midfix_causal_features = Activation('relu')(midfix_causal_features)

    postfix_causal_features = Convolution1D(filters=80, kernel_size=5, activation='relu', padding='same')(
        postfix_causal_input)
    postfix_causal_features = Dense(16, use_bias=False)(postfix_causal_features)
    postfix_causal_features = BatchNormalization()(postfix_causal_features)
    postfix_causal_features = Activation('relu')(postfix_causal_features)

    combined_causal_features = concatenate([prefix_causal_features, midfix_causal_features, postfix_causal_features], axis=-1)

    pos_features = Embedding(input_dim=50, output_dim=16,
                             input_length=(maxlen_seq, ), embeddings_regularizer=tf.keras.regularizers.l2(.001))(pos_input)
    pos_features = Bidirectional(LSTM(20, return_sequences=True, activation='tanh',
                                       dropout=0.2, recurrent_dropout=0.2), merge_mode='ave')(pos_features)

    word_and_causal_features = concatenate([word_features, combined_causal_features], axis=-1)
    word_and_causal_features = MultiHeadAttention(head_num=int(word_and_causal_features.shape[2]))(word_and_causal_features)

    pos_and_causal_features = concatenate([pos_features, combined_causal_features], axis=-1)
    pos_and_causal_features = MultiHeadAttention(head_num=int(pos_and_causal_features.shape[2]))(
        pos_and_causal_features)

    merged_features = concatenate([word_and_causal_features, pos_and_causal_features], axis=-1)
    predictions = Dense(nclasses, activation='softmax')(merged_features)
    model = Model(inputs=[word_input, prefix_causal_input, midfix_causal_input, postfix_causal_input, pos_input], outputs=predictions)

    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Set up callbacks
    fileprefix = 'embed_fixed_'
    am = approximateMatch.ApproximateMatch_SEQ(valid_toks, valid_y, valid_xs, idx2label,
                                               pred_dir=os.path.join(basedir, 'predictions'), fileprefix=fileprefix)

    mc = callbacks.ModelCheckpoint(os.path.join(basedir, 'models', 'embedfixed.model.weights.{epoch:02d}.hdf5'))
    th = TimeHistory()
    cbs = [am, mc, th]
    if validate:
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3)
        cbs.append(early_stopping)

    # Train the model
    print('Training...')
    hist = model.fit(train_xs, train_y, epochs=num_epochs, batch_size=1,
                     validation_data=(valid_xs, valid_y), callbacks=cbs)

    if validate:
        val_f1, best_model = learning_curve(hist, preddir=os.path.join(basedir, 'predictions'),
                                            pltname=os.path.join(basedir, 'charts', 'history.pdf'), fileprefix=fileprefix)
    else:
        best_model = num_epochs - 1
        val_f1 = 0.0

    # Save model
    json_string = model.to_json()
    open(os.path.join(basedir, 'models', 'embedfixed_model_architecture.json'), 'w').write(json_string)
    if best_model == 0:
        best_model = len(hist.history['val_loss'])
    # Test
    bestmodelfile = os.path.join(basedir, 'models', 'embedfixed.model.weights.%02d.hdf5' % best_model)
    shutil.copyfile(bestmodelfile, bestmodelfile.replace('.hdf5', '.best.hdf5'))
    if validate:
        custom_objects = {'MultiHeadAttention': MultiHeadAttention}
        model = model_from_json(open(os.path.join(basedir, 'models', 'embedfixed_model_architecture.json')).read(),
                                custom_objects)
        model.load_weights(bestmodelfile)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    scores = predict_score(model, test_xs, test_toks, test_y, os.path.join(basedir, 'predictions'), idx2label,
                           maxlen_seq, fileprefix=fileprefix)

    scores['val_f1'] = val_f1
    scores['train_time'] = sum(th.times)
    return scores, hist.history, best_model

def build_directory_structure(basedir):
    try:
        os.makedirs(basedir)
    except:
        pass
    try:
        os.makedirs(os.path.join(basedir, 'charts'))
    except:
        pass
    try:
        os.makedirs(os.path.join(basedir, 'models'))
    except:
        pass
    try:
        os.makedirs(os.path.join(basedir, 'predictions'))
    except:
        pass
    try:
        os.makedirs(os.path.join(basedir, 'histories'))
    except:
        pass
    try:
        os.makedirs(os.path.join(basedir, 'preprocessing'))
    except:
        pass
    try:
        os.makedirs(os.path.join(basedir, 'results'))
    except:
        pass


if __name__ == "__main__":
    optparser = optparse.OptionParser()
    optparser.add_option("-b", "--basedir", dest="basedir", default='../model_output',
                         help="Base directory for model output files")
    optparser.add_option("-P", "--picklefile", dest="picklefile", default=SETTINGS['PICKLEFILE'],
                         help="Pickle file containing twitter dataset (output from prep.py)")
    optparser.add_option("-s", "--seedrand", dest="seedrand", type="int", default=1)
    optparser.add_option("-e", "--nbepochs", dest="nbepochs", type="int", default=30)
    optparser.add_option("-m", "--model", dest="model", default='scan')
    (opts, _) = optparser.parse_args()

    ###### ******* Set seed values for reproducible results
    seed_value = opts.seedrand
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.set_random_seed(seed_value)
    ##### ****** End of setting seed values

    build_directory_structure(opts.basedir)

    ## Load initial embedding weights
    sys.stderr.write('Loading embeddings...\n')

    model_path = SETTINGS['W2VEC_TWTR_MODEL_PATH']
    w2v = w2vec.Word2Vec.load_word2vec_format(model_path, binary=True)
    embed_dim = w2v.syn0norm.shape[1]


    ## Load the data
    train_set, valid_set, test_set, dic = prep.load_adefull(opts.picklefile)
    idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
    idx2word = dict((k, v) for v, k in dic['words2idx'].items())

    char_vocab = set()
    for word in idx2word.values():
        char_vocab |= set(list(word.lower()))
    char_vocab = sorted(list(char_vocab))

    idx2char = dict((k, v) for k, v in enumerate(char_vocab))
    idx2char[len(idx2char.keys())] = '<u>'
    idx2char[len(idx2char.keys())] = '<p>'

    if 0 in idx2label:
        sys.stderr.write('Index 0 found in labels2idx: data may be lost because 0 used as padding\n')
    if 0 in idx2word:
        sys.stderr.write('Index 0 found in words2idx: data may be lost because 0 used as padding\n')
    idx2word[0] = 'PAD'
    idx2label[0] = 'PAD'

    train_toks, train_lex, train_y = train_set
    valid_toks, valid_lex, valid_y = valid_set
    test_toks, test_lex, test_y = test_set

    vocsize = max(idx2word.keys()) + 1
    nclasses = max(idx2label.keys()) + 1

    dataset = (train_toks, valid_toks, test_toks, train_lex, valid_lex, test_lex, train_y, valid_y, test_y)
    if len(valid_lex) > 0:
        validate = True
    else:
        validate = False

    if opts.model == 'scan':
        scores, history, best_model = model_scan(dataset, idx2word, idx2label, w2v, opts.basedir,
                                                 validate=validate, num_epochs=opts.nbepochs)
    else:
        print('Invalid model')
        exit()
    ## Retrieve scores
    if validate:
        val_loss = history['val_loss'][best_model-1]
        val_f1 = scores['val_f1']

    training_loss = history['loss'][best_model-1]
    train_time = scores['train_time']
    pred_time = scores['pred_time']
    test_f1 = scores['f1']
    test_loss = scores['loss']
    print('Scores for training set size: %d\n'
          '--------------------------\n'
          'training loss %0.4f\n'
          % (len(train_lex), training_loss))
    if validate:
        print('validation loss %0.4f\n'
              'validation f1 %0.4f\n'
              % (val_loss, val_f1))
    print('test loss %0.4f\n'
          'test f1 %0.4f\n'
          % (test_loss, test_f1))

    print('Training time: {:.2f}s ({:.4f}s/tweet)\n'
          'Prediction time: {:.2f}s ({:.4f}s/tweet)'.format(train_time,
                                                            train_time/len(train_lex),
                                                            pred_time,
                                                            pred_time/len(test_lex)))

    result_file = os.path.join(opts.basedir, 'results', 'adr_label_results.txt')
    with open(result_file, 'a+') as f:
        result = str(scores['p']) + ',' + str(scores['r']) + ',' + str(scores['f1']) + '\n'
        f.write(result)
