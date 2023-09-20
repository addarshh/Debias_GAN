'''
Process Tweets
Tokenize texts
Reduce corpus, based on the sample dataset
Set sequence length
'''

from transformers import AutoModel, AutoTokenizer
import emoji
import numpy
import re
import json
import pandas as pd
import itertools

# process emoji
def parse_surrogate_pairs(line):
    regex_1 = r"u[a-zA-Z0-9]{4}"
    x_1 = re.findall(regex_1,line)
    x_2 = [item for item in x_1 if len(re.findall(r"\d",item))>0]
    list_1 = ''.join(x_2)
    list_2 = '\\'+'\\'.join(x_2)
    return(list_1,list_2)

def surrogate_to_emoji(list_2):    
    emoji_ele = json.loads('"'+list_2+'"')
    return(emoji_ele)

def replace_surrogate_with_emoji(line,list_1,emoji_ele):
    return(line.replace(list_1,emoji_ele))

def preprocess_tweet(ele):
    list_1,list_2 = parse_surrogate_pairs(ele)
    try:
        emoji_ele = surrogate_to_emoji(list_2)
        new_ele = replace_surrogate_with_emoji(ele,list_1,emoji_ele)
    except:
        new_ele = ele
    return(new_ele)

# process html symbol
# replace some symbols with actual symbols: &lt;<, &amp;&, &gt;>
def html_symbol(ele):
    return(ele.replace('&lt;','<').replace('&gt','>').replace('&amp;','&'))

# tokenizer
def tokenize_tweet(tweet_list):
    bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base",use_fast = False, normalization = True)
    test_tweet_encoded = tokenizer(tweet_list,padding=True, truncation=True)
    return(test_tweet_encoded['input_ids'])

# use a smaller corpus
def new_corpus(tweet_list):
    flat = set(itertools.chain(*tweet_list))
    new_vocab = pd.DataFrame(flat,columns=['original_idx']).reset_index()
    corpus_key = new_vocab['original_idx']
    corpus_val = new_vocab['index']
    # save the lookup
    new_vocab.to_csv('new_vocab.csv',header = True, index = False)
    corpus_lookup = {corpus_key[i]: corpus_val[i] for i in range(len(corpus_key))}
    return(corpus_lookup)

# replace with smaller corpus
def reduced_token(ele,corpus_lookup):
    new_ele = [corpus_lookup[idx] for idx in ele]
    return(new_ele)

# sequence length distribution (without 1)
def sequence_length(ele):
    return(sum([i!=1 for i in ele])-2)

# read data
tweet_df = pd.read_csv('tweet_data_full.csv')

# process text 
# emoji
tweet_df['process_1'] = tweet_df['text'].apply(preprocess_tweet)
tweet_df['process_2'] = tweet_df['process_1'].apply(html_symbol)

# tokenize
tweet_df['token'] = tokenize_tweet(tweet_df['process_2'])

# sequence length
tweet_df['seq_length'] = tweet_df['reduced_token'].apply(sequence_length)

# use reduced sample
tweet_sample_df = tweet_df.loc[tweet_df['seq_length']==10].copy()

# get reduced corpus
# use reduced corpus
corpus_lookup = new_corpus(tweet_sample_df['token'])
tweet_sample_df['reduced_token'] = tweet_sample_df['token'].apply(reduced_token,corpus_lookup = corpus_lookup)

# write to a txt file for model consumption
tweet_list = tweet_sample_df['reduced_token'].to_list()
with open('tweets_seq_10.txt', 'w') as f:
    for item in tweet_list:
        item = str(item[1:11]).replace(',','').replace('[','').replace(']','')
        f.write("%s\n" % item)


tweet_df[['text','mention','race','token','seq_length']].to_csv('processed_tweet_df.csv',header = True, index = False)
