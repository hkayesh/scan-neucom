SETTINGS = {
    'PICKLEFILE': './data/seq_labeling/processed/ade.asu_chop.ioadrind.valid010.pkl',
    'SEQLAB_DATA_DIR': './data/seq_labeling',
    'DYLD_FALLBACK_LIBRARY_PATH': '$DYLD_FALLBACK_LIBRARY_PATH:~/anaconda/lib/',

    ## Path to word2vec twitter model. YOU MUST DOWNLOAD THIS separately - see
    ##  ./word2vec_twitter_model/readme_w2v for the URL
    'W2VEC_TWTR_MODEL_PATH': 'word2vec_twitter_model/word2vec_twitter_model.bin',
    # 'W2VEC_TWTR_MODEL_PATH': '../ADR/word2vec_twitter_model/word2vec_twitter_model.bin',

    # 'BERT_EMBD_MODEL_PATH': '/home/humayun/bert-word-embeddings/bert-base-uncased.30522.768d.vec',  # downloaded from https://github.com/Kyubyong/bert-token-embeddings
    # 'BERT_EMBD_MODEL_PATH': '/home/humayun/bert-word-embeddings/bert-base-cased.28996.768d.vec',  # downloaded from https://github.com/Kyubyong/bert-token-embeddings
    # 'BERT_EMBD_MODEL_PATH': '/home/humayun/bert-word-embeddings/bert-large-uncased.30522.1024d.vec',  # downloaded from https://github.com/Kyubyong/bert-token-embeddings
    # 'BERT_EMBD_MODEL_PATH': '/home/humayun/bert-word-embeddings/bert-large-cased.28996.1024d.vec',  # downloaded from https://github.com/Kyubyong/bert-token-embeddings

    ## Path to Twitter dataset files containing Tweet IDs and Annotations
    # 'ASU_DATASET': './data/asu_tweets/download_tweets',
    # 'CHOP_DATASET': './data/chop_adhd_adr_supp',

    ## The B-I-O labeling scheme
    'LABELSET': {'ADR': {'b': 'I-ADR', 'i': 'I-ADR'},
                 # 'Indication': {'b': 'I-Indication', 'i': 'I-Indication'},
                 # 'Beneficial': {'b': 'I-Indication', 'i': 'I-Indication'},
                 'Indication': {'b': 'O', 'i': 'O'},
                 'Beneficial': {'b': 'O', 'i': 'O'}
                 }
}