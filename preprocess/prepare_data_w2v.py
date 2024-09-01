from datasets import load_dataset,DatasetDict,load_from_disk
from gensim.models import KeyedVectors
import random
import re
import pandas as pd
import sys
save_path = './data/'

def map_labels(label):
    if label in [0, 1]:
        return 0
    elif label in [3, 4]:
        return 1
    else:
        return -1
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'([,.!?()])', ' ', text)
    text = ' '.join(text.split()).lower()
    return text

for dataset in ['SetFit/sst5','emotion', 'rotten_tomatoes', 'hate']:
    dataname = dataset
    if dataname in ['emoji', "sentiment", "stance_abortion", "stance_atheism", "stance_climate", "stance_feminist", \
                "stance_hillary", 'hate','emotion']:
        dataset = load_dataset('tweet_eval', dataname)
    else:
        dataset = load_dataset(dataname)
    
    df_trian = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()
    if dataname=='SetFit/sst5':
        df_trian = df_trian[df_trian['label'].apply(map_labels)!=-1]
        df_trian['label'] = df_trian['label'].apply(map_labels)
        df_test = df_test[df_test['label'].apply(map_labels)!=-1]
        df_test['label'] = df_test['label'].apply(map_labels)
    else:
        df_trian = df_trian[df_trian['label'].apply(lambda x: x in [0,1])]
        df_test = df_test[df_test['label'].apply(lambda x: x in [0,1])]

    df_trian['exp_split'] = "train"
    df_test['exp_split'] = "test"
    
    print("train set number:", len(df_trian))
    print("test set number:", len(df_test))
    df = df_trian._append(df_test)
    df = df.dropna(axis=0,how='any')
    df_test = df_test.dropna(axis=0,how='any')
    df_trian = df_trian.dropna(axis=0,how='any')
    if dataname == "SetFit/sst5":
        dataname = "sst"
        
    import os
    df_path = f'./{dataname}'
    try:
        os.mkdir(df_path)
    except BaseException as e:
        print(e)
    df_file = f"./{dataname}/{dataname}.csv"
    df.to_csv(df_file)

    import argparse
    parser = argparse.ArgumentParser(description='Run Preprocessing on dataset')
    parser.add_argument('--data_file', type=str,default='./data/emotion/emotion.csv')
    parser.add_argument("--output_file", type=str,default="./data/emotion/vec.p")
    parser.add_argument('--word_vectors_type', type=str, choices=['fasttext.simple.300d'], default="fasttext.simple.300d")
    parser.add_argument('--min_df', type=int,default=1)
    parser.add_argument(
            '-f',
            '--file',
            help='Path for input file. First line should contain number of lines to search in'
        )
    args, extras = parser.parse_known_args()
    args.extras = extras

    args.data_file = df_file
    args.output_file = os.path.join(df_path,'vec.p')
    import vectorizer
    vec = vectorizer.Vectorizer(min_df=args.min_df)
    assert 'text' in df.columns, "No Text Field"
    assert 'label' in df.columns, "No Label Field"
    assert 'exp_split' in df.columns, "No Experimental splits defined"
    texts = list(df[df.exp_split == 'train']['text'])
    vec.fit(texts)
    print("Vocabulary size : ", vec.vocab_size)
    vec.seq_text = {}
    vec.label = {}
    vec.raw_text = {}
    splits = df.exp_split.unique()
    for k in splits :
        split_texts = list(df[df.exp_split == k]['text'])
        vec.raw_text[k] = split_texts
        vec.seq_text[k] = vec.get_seq_for_docs(split_texts)
        vec.label[k] = list(df[df.exp_split == k]['label'])
    if args.word_vectors_type in ['fasttext.simple.300d'] :
        vec.extract_embeddings_from_torchtext(args.word_vectors_type,cache="./.vector_cache")
    else :
        vec.embeddings = None
    
    import pickle, os
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    pickle.dump(vec, open(args.output_file, 'wb')) 
