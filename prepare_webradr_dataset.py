import re
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split

# dataset source url: https://link.springer.com/article/10.1007/s40264-020-00912-9

dataset_file_path = 'data/seq_labeling/raw/webadr_dataset_AE_tweets.csv'
out_file_full_path = 'data/seq_labeling/raw/webadr_full_raw.csv'
out_file_train_path = 'data/seq_labeling/raw/train/webadr_75_25_train'
out_file_test_path = 'data/seq_labeling/raw/test/webadr_75_25_test'

df = pd.read_csv(dataset_file_path)
new_df = pd.DataFrame(columns=['tweet_id', 'begin', 'end', 'type', 'extraction', 'drug', 'tweet', 'meddra_code', 'meddra_term'])
# 332317478170546176,28,37,ADR,allergies,avelox,"do you have any medication allergies? ""asthma!!!"" me: ""........"" pt: ""no wait. avelox, that's it!"" ""so no other allergies?"" ""right!"" *cont",10013661,drug allergy

data_list = []

for row in df.to_dict(orient='row'):
    tweet = row['Tweets']
    if not isinstance(tweet, str):
        continue  # skipping tweets that has no text

    tweet = row['Tweets'].strip()
    tweet = re.sub('[\n\r\t]', '', tweet)
    drugs = row['Product(s) as reported'].strip().split(';')
    if len(drugs) > 1:
        continue  # excluding the tweets with more than one drug names [total 10 such tweets]

    adrs = row['Event(s) as reported'].strip().split(';')
    anno_type = 'ADR'
    for adr in adrs:
        try:
            match = re.search(adr, tweet)
            begin = match.start()
            end = match.end()
        except Exception:
            continue  # excluding tweets if ADR text does not exist in the tweet
        data = {
            'tweet_id': row['Twitter ID'],
            'begin': match.start(),
            'end': match.end(),
            'type': anno_type,
            'extraction': adr,
            'drug': row['Product(s) as reported'],
            'tweet': tweet,
            'meddra_code': None,
            'meddra_term': None
        }
        data = OrderedDict(data.items())
        data_list.append(data)
data_df = pd.DataFrame(data_list)
data_df.to_csv(out_file_full_path, index=None)

train_df, test_df = train_test_split(data_df, test_size=0.25, random_state=1)
train_df.to_csv(out_file_train_path, index=None)
test_df.to_csv(out_file_test_path, index=None)

print(train_df.shape)
print(train_df.loc[data_df['type'] == 'ADR'].shape)
print(test_df.shape)
print(test_df.loc[data_df['type'] == 'ADR'].shape)


