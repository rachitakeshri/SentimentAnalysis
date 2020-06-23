from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier, DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

x_train,y_train,x_test,y_test=[],[],[],[]
import re, string, random



'''
     This function is responsible for cleaing the dataset by:
     1) tokenization, 
     2) filtering (urls etc), 
     3) part of speech tagging
     4) lemmatization 
'''
def remove_noise(tweet_tokens, stop_words = ()):
  
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def get_models():

    #This function defines the four models we will be using for the classification purpose

    models=dict()
    models['nb'] = MultinomialNB()
    models['logreg'] = LogisticRegression(solver='liblinear',multi_class='auto',C=1)
    models['rf'] = RandomForestClassifier(n_estimators=500)
    models['lsvm'] = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=1e-3)

    return models

def evaluate_model(model):
    model.fit(X_train_count, y_train)
    y_pred = model.predict(X_test_count)
    return(accuracy_score(y_pred, y_test))

def custom_tweet(tweet):
    classifier = NaiveBayesClassifier.train(dataset[:10000])

    
    return (classifier.classify(tweet))

    


if __name__ == "__main__":

    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
   # print(positive_tweets[:10])
   # print(negative_tweets[:10])
   # Uncomment above lines of see samples of tweets

   # text = twitter_samples.strings('tweets.20150430-223406.json')
   #tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive")
                         for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                         for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset



    random.shuffle(dataset)


    x_train=[row[0] for row in dataset[:7000]]
    y_train=[row[1] for row in dataset[:7000]]
    
    x_test=[row[0] for row in dataset[7000:]]
    y_test=[row[1] for row in dataset[7000:]]

#We are using DictVectorizer to transform the list of feature dictionary into numerical form or simply for word embedding

    from sklearn.feature_extraction import DictVectorizer
    d = DictVectorizer()
    d.fit([row[0] for row in dataset])
    X_train_count =  d.transform(x_train)

    X_test_count =  d.transform(x_test)



    models=get_models()

    for name,model in models.items():

            scores=evaluate_model(model)
            #results.append(scores)
            print(name,scores)






    custom_twee = "Tornado destroyed the buildings in the whole city and killed dozens of people."

    custom_tokens = remove_noise(word_tokenize(custom_twee))

    c=dict([token, True] for token in custom_tokens)

    print(custom_twee,"  ",custom_tweet(c))
