from datasets import load_dataset,DatasetDict,load_from_disk
save_path = './data/'
def map_labels(label):
    if label in [0, 1]:
        return 0
    elif label in [3, 4]:
        return 1
    else:
        return -1
for dataset in ['emotion', 'SetFit/sst5', 'rotten_tomatoes', 'hate']:
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
    df = df_trian._append(df_test)
    df = df.dropna(axis=0,how='any')
    print('token mean length:',df['text'].apply(lambda x: len(x)).mean())
    print("train data negative:",df_trian['label'].count()-df_trian['label'].sum())
    print("train data positive:",df_trian['label'].sum())
    print("test data negative:",df_test['label'].count()-df_test['label'].sum())
    print("test data positive:",df_test['label'].sum())
    if dataname == "SetFit/sst5":
        dataname = "sst"
    import os
    df_path = f'./{dataname}'
    try:
        os.mkdir(df_path)
    except:
        pass
    df_file = f"./{dataname}/data.csv"
    df.to_csv(df_file)
