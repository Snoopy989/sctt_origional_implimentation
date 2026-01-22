"""
CAVEATS:
- no spell checking as of now
- no standardization of rater-wise basis for severity
- currently doesn't do stratefied sampling of the prompts--it's just random subset
"""
import pandas as pd
from datasets import Dataset, Features, ClassLabel, Value, load_metric, DatasetDict
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#  LOAD TOOLS
scaler = StandardScaler()
mmscaler = MinMaxScaler()

def preprocess_llm_data(d, gen, val_pct, val_train_pct, test_pct, prefix, connector1, connector2):

    #  PROCESS DUPES
    dupes = d[d.duplicated(subset = ['item','task', 'prompt','response'], keep = False)]  # select out the dupes
    dupes.to_csv('d_dupes.csv', index = False)
    dupes[['study_id', 'sub_id', 'response_id', 'ID']] = 'dupe_average'
    print(dupes.head())
    dupes_avg = dupes.groupby(['item', 'task', 'prompt', 'response'], as_index=False).agg({'study_id': 'first', 'sub_id': 'first', 'response_id': 'first', 'ID': 'first', 'jrt': 'mean'}).reset_index(drop = True)  # get avg jrt across dupe'd responses
    print(dupes_avg.head())
    dupes_avg.to_csv('d_dupes_avg.csv', index = False)
    d = d.drop(list(dupes.index), axis = 0)  # drop all items at the dup'd indices
    d = pd.concat([d, dupes_avg]).reset_index(drop = True)  # combine avg dupes with filtered df

    #  RENAME/adjust VARS
    d['text'] = prefix+d['task']+connector1+d['prompt']+connector2+d['response'].str.lower()
    d['text'] = d['text'].astype(str)
    mmscaler = MinMaxScaler()
    d['jrt'] = d['jrt']+abs(d.jrt.min())
    d['label'] = mmscaler.fit_transform(d['jrt'].to_numpy().reshape(-1,1))
    d = d[['ID', 'item', 'prompt', 'response', 'jrt', 'text', 'label']]
    d.dropna()
    print(d.info())
    d.to_csv('sctt_jrt_cleaned.csv', index = False)

    #  SETUP held out item generalization set
    print(gen.head())
    gendupes = gen[gen.duplicated(subset = ['prompt','response'], keep = False)]  # select out the gendupes (NONE EXIST)
    gen['text'] = prefix+gen['task']+connector1+gen['prompt']+connector2+gen['response'].str.lower()
    gen['text'] = gen['text'].astype(str)
    mmscaler = MinMaxScaler()
    gen['jrt'] = gen['jrt']+abs(gen['jrt'].min())
    gen['label'] = mmscaler.fit_transform(gen['jrt'].to_numpy().reshape(-1,1))
    heldout = gen
    heldout['ID'] = '{}-{}'.format(heldout['study_id'], str(heldout['response_id']))
    heldout = heldout[['ID', 'item', 'prompt', 'response', 'text', 'label']]
    heldout.dropna()
    print(heldout.info())

    #  TRAIN/VAL/TEST SPLIT
    features = Features({"ID": Value("string"),
                         "item": ClassLabel(num_classes = len(list(d.item.unique())), names = list(d.item.unique())),
                         "prompt": ClassLabel(num_classes = len(list(d.prompt.unique())), names = list(d.prompt.unique())),
                         "response": Value("string"),
                         "jrt": Value("float32"),
                         "text": Value("string"),
                         "label": Value("float32")})  # define dataset features
    features_heldout = Features({"ID": Value("string"),
                                 "item": ClassLabel(num_classes = len(list(gen.item.unique())), names = list(gen.item.unique())),
                                 "prompt": Value("string"),
                                 "response": Value("string"),
                                 "text": Value("string"),
                                 "label": Value("float32")})  # define dataset features
    dataset = Dataset.from_pandas(d, preserve_index = False, features = features) # import to datasets via pandas
    heldout_dataset = Dataset.from_pandas(heldout, preserve_index = False, features = features_heldout) # import to datasets via pandas
    dataset = dataset.train_test_split(test_size = test_pct, shuffle = True, seed = 42)  # split dataset in stratefied way (by prompt)
    dataset_val = dataset['train'].train_test_split(test_size = val_train_pct, shuffle = True, seed = 42)
    train_val_test_dataset = DatasetDict({
        'train': dataset_val['train'],
        'validation': dataset_val['test'],
        'test': dataset['test'],
        'heldout': heldout_dataset
    })

    ## SAVE CSV COPIES OF DATA FOR SEMDIS/ELABORATION
    dataset_val['train'].to_csv('training_items_sctt.csv')
    dataset_val['test'].to_csv('validation_items_sctt.csv')
    dataset['test'].to_csv('test_items_sctt.csv')
    heldout_dataset.to_csv('heldoutprompt_items_sctt.csv')

    #  REMOVE UNECESSARY COLUMNS
    for i in train_val_test_dataset:
        if i != "heldout":
            train_val_test_dataset[i] = train_val_test_dataset[i].remove_columns(["ID","item","prompt", "response","jrt"])
        else:

            train_val_test_dataset[i] = train_val_test_dataset[i].remove_columns(["ID","item","prompt", "response"])
    return train_val_test_dataset
