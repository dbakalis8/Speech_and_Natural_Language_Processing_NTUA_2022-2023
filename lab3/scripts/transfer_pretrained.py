from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils.load_datasets import load_MR, load_Semeval2017A
from training import get_metrics_report

# DATASET = 'MR'
# PRETRAINED_MODELS = ['siebert/sentiment-roberta-large-english', 'sohan-ai/sentiment-analysis-model-amazon-reviews', 'distilbert-base-uncased-finetuned-sst-2-english']

DATASET = 'Semeval2017A'
PRETRAINED_MODELS = ['cardiffnlp/twitter-roberta-base-sentiment', 'finiteautomata/bertweet-base-sentiment-analysis', 'hakonmh/sentiment-xdistil-uncased']


LABELS_TOKENIZER_MAPPING = {
    'siebert/sentiment-roberta-large-english': {
        'LABELS':{
            'NEGATIVE': 'negative',
            'POSITIVE': 'positive'
         },
         'TOKENIZER': None
    },

    'sohan-ai/sentiment-analysis-model-amazon-reviews': {
        'LABELS': {
            'LABEL_0': 'negative',
            'LABEL_1': 'positive'
        },
        'TOKENIZER': 'distilbert-base-uncased'
    },

    'distilbert-base-uncased-finetuned-sst-2-english': {
        'LABELS':{
            'NEGATIVE': 'negative',
            'POSITIVE': 'positive'
         },
         'TOKENIZER': None
    },

    'cardiffnlp/twitter-roberta-base-sentiment': {
        'LABELS':{
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral',
            'LABEL_2': 'positive'
        },
        'TOKENIZER': None
    },

    'finiteautomata/bertweet-base-sentiment-analysis': {
        'LABELS':{
            'NEG': 'negative',
            'NEU': 'neutral',
            'POS': 'positive'
        },
        'TOKENIZER': None
    },

     'hakonmh/sentiment-xdistil-uncased': {
        'LABELS':{
            'Negative': 'negative',
            'Neutral': 'neutral',
            'Positive': 'positive'
        },
        'TOKENIZER': None
    }
    
}

if __name__ == '__main__':
    # load the raw data
    if DATASET == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
    elif DATASET == "MR":
        X_train, y_train, X_test, y_test = load_MR()
    else:
        raise ValueError("Invalid dataset")

    # encode labels
    le = LabelEncoder()
    le.fit(list(set(y_train)))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    n_classes = len(list(le.classes_))

    #for model in PRETRAINED_MODELS: 
    model = PRETRAINED_MODELS[2]
        # define a proper pipeline
    if LABELS_TOKENIZER_MAPPING[model]['TOKENIZER'] == None:
        sentiment_pipeline = pipeline("sentiment-analysis", model=model)  
    else:
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=LABELS_TOKENIZER_MAPPING[model]['TOKENIZER'])

    y_pred = []
    
    for x in tqdm(X_test):
        # TODO: Main-lab-Q6 - get the label using the defined pipeline 
        label = sentiment_pipeline(x)[0]['label']  
        y_pred.append(LABELS_TOKENIZER_MAPPING[model]['LABELS'][label])


    y_pred = le.transform(y_pred)
    print(f'\nDataset: {DATASET}\nPre-Trained model: {model}\nTest set evaluation\n{get_metrics_report([y_test], [y_pred])}')