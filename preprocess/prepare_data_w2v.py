#if you want to get the result in the last form in our paper, please modify the comment section of this code.
#we advise you to preturb the data manually,since the result will be better.
#This is my first time writing a paper, so there may be many flaws in the code. 
#If you find any errors or issues, please let me know and I would greatly appreciate it. 
from datasets import load_dataset,DatasetDict,load_from_disk
from gensim.models import KeyedVectors
import random
import re
import pandas as pd
import sys
save_path = './data/'

#this func is used to classify sst5. In this dataset,negative data's tags are 0/1, and the positive's are 3/4.
def map_labels(label):
    if label in [0, 1]:
        return 0
    elif label in [3, 4]:
        return 1
    else:
        return -1

#this func is used to preprocess the sentence so that it can be split into words easily.
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'([,.!?()])', ' ', text)
    text = ' '.join(text.split()).lower()
    return text

for dataset in ['SetFit/sst5','emotion', 'rotten_tomatoes', 'hate']:
    dataname = dataset
    #if you have local dataset:
    #dataset = load_from_disk(save_path+dataname)
    #else:
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
    #these commented out code part is used to preturb the data. It replace 2 random words in the sentence with similar one.
    #however,we still advise you to change it manually.
    '''i=0
    model_path='./.vector_cache/wiki.simple.vec'
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=False)
    if word_vectors['word'] is not None:
        print("gensim load successed！")
    else:
        print("gensim load failed.")
    modified_testdata=[]
    for data in df_test['text']:
        sentence_raw=data
        i+=1
        sentence=clean_text(sentence_raw)
        words = sentence.split()  
        """ if i<100:
            print(i,'.',sentence_raw, '  ',words[index1],'  ',words[index2]) """
        for _ in range(5):
            index2 = -1
            index1 = random.randint(0, len(words) - 1)
            while index1 == index2:
                index2 = random.randint(0, len(words) - 1)
                if len(sentence) == 1:
                    break
            try:
                words1 = str(word_vectors.most_similar(words[index1], topn=1)[0][0])
                words2 = str(word_vectors.most_similar(words[index2], topn=1)[0][0])
            except:
                continue
            if not (words1 is None or words2 is None):
                new_sentence = str(sentence_raw.replace(words[index1], words1, 1).replace(words[index2], words2, 1))
                modified_testdata.append(new_sentence)
                break
        else:
            modified_testdata.append(sentence_raw)

        if modified_testdata[-1] is None:
            modified_testdata[-1] = data
            print(sentence_raw)
        """ if i<100:
            print(new_sentence) """
    print(len(modified_testdata))
    modified_testdata={'text':modified_testdata}
    modified_testdata=pd.DataFrame(modified_testdata)
    if modified_testdata.isnull().any().any():
        print("error")
        sys.exit(1)
    df_test['text'] = modified_testdata['text'].apply(str)'''
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

    #from attention.preprocess 
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