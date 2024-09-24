
################################################################
# 1. Developing Recommendations Based on Coffeeshop Reviews_Text
################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud

from string import digits

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df = pd.read_csv(r'C:\Users\ozgur\OneDrive\Desktop\Project\raw_yelp_review_data.csv', encoding='utf-8', delimiter=',', skipinitialspace=True)

df.head()
df.shape

#Delete the duplicates:

df.loc[df["coffee_shop_name"] == "Caffé Medici ", ["coffee_shop_name"]] = "Caffe Medici "
df.loc[df["coffee_shop_name"] == "Lola Savannah Coffee Downtown ", ["coffee_shop_name"]] = "Lola Savannah Coffee Lounge "
df.loc[df["coffee_shop_name"] == "Summer Moon Coffee Bar ", ["coffee_shop_name"]] = "Summermoon Coffee Bar "

df.head()

#Delete all numbers including the date
for i in df.index :
    remove_digits = str.maketrans('', '', digits)
    df["full_review_text"][i] = df["full_review_text"][i].translate(remove_digits)
    df["full_review_text"][i] = df["full_review_text"][i].lstrip(df["full_review_text"][i][0:3])
    #stringlerde sondan başlayıp eleman silme
    df["star_rating"][i] = df["star_rating"][i].rstrip(df["star_rating"][i][3:])


#Convert ratings to float:
df["star_rating_num"] = df["star_rating"].astype(float)

#Delete spaces:
df['coffee_shop_name'] = df['coffee_shop_name'].str.strip()

#Make all comments lowercase
df['full_review_text'] = df['full_review_text'].str.lower()

#Before substituting this df into "group by" and simplifying, it should have been done; it is necessary for visualization
df_wordcloud = df.copy()

#Delete stop words
stop_words = ["check-in", "check-ins"] + stopwords.words("english")
df["full_review_text"] = df["coffee_shop_description"] = df["full_review_text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

#Delete punctuation
df["coffee_shop_description"] = df["coffee_shop_description"].str.replace(r'[^\w\s]+', '')
df1 = df.copy()  #Copied without applying 'group by' for visualization

#I am creating a new variable called 'coffe_shop_description' and saving the new dataframe again as 'df' to avoid variable redundancy.
#The old dataframe is disappearing, but it's not needed anyway.  #New 'df' to be used in TF-IDF
df = df.groupby(["coffee_shop_name"]).agg({"coffee_shop_description": "sum"}).reset_index()

df.shape  #(76, 2)
df.head(76)



#################################################
# 2. The creation of the TF-IDF matrix
#################################################

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df['coffee_shop_description'])

tfidf_matrix.shape
#(76, 25168)
#Our comments are in the rows, while unique words are in the columns.

#All features in the columns have been received
tfidf.get_feature_names()

#Let's score the words / features in these columns
#Scores at the intersections of documents and terms
tfidf_matrix.toarray()


#################################################
# 3. Creation of the Cosine Similarity Matrix
#################################################

#With this matrix, the similarity of each café with the other cafés is represented
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


cosine_sim.shape  #(76, 76)
cosine_sim[1]
cosine_sim[5]

#################################################
# 4. Making Café Recommendations Based on Similarities
#################################################

indices = pd.Series(df.index, index=df['coffee_shop_name'])
indices.index.value_counts()

#Let's remove duplications in café shop names, according to the latest comments
indices = indices[~indices.index.duplicated(keep='last')]

indices["Caffe Medici"]

#
cafe_index = indices["Caffe Medici"]
cosine_sim[cafe_index]


similarity_scores = pd.DataFrame(cosine_sim[cafe_index], columns=["score"])

cafe_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

df['coffee_shop_name'].iloc[cafe_indices]



#################################################
## 4. Data Analysis & Visualization
#################################################

###The analysis of numerical data (single numerical data: star_rating_num)
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.ylabel("count")
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df1, "star_rating_num", plot=True)

### heat map :
corr = pd.DataFrame(cosine_sim)
sns.heatmap(corr.corr(), xticklabels=True, yticklabels=True, cmap="viridis", vmin=-1, vmax=1, center= 0, square=True)

plt.show()


#The distribution of scores given by individuals - Pie Chart

df1.groupby(['star_rating']).sum().plot(
	kind='pie', y='star_rating_num', autopct='%1.0f%%')
plt.show()


#Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()

#Apply lambda function to get compound scores.
function = lambda title: vader.polarity_scores(title)['compound']
df1['compound'] = df1['coffee_shop_description'].apply(function)
df1.head(5)

#Word cloud visualization
from wordcloud import WordCloud

allWords = ' '.join([twts for twts in df_wordcloud['full_review_text']])

wordCloud = WordCloud(colormap= "flare", background_color="white", contour_color ="yellow",
                      width=500, height=300, random_state=21, max_font_size=110).generate(allWords)

plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()


def getAnalysis(score):
 if score < 0:
    return 'Negative'
 elif score == 0:
    return 'Neutral'
 else:
    return 'Positive'

df1['coffee_shop_description'] = df1['compound'].apply(getAnalysis)

df1.head(5)

#Visualize the counts for each sentiment type.
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df1['coffee_shop_description'].value_counts().plot(kind = 'bar')
plt.show()

#pie chart:
df1.coffee_shop_description.value_counts().plot(kind='pie', autopct='%1.0f%%',  fontsize=8, figsize=(9,6), colors=["purple", "yellow", "pink"])
plt.ylabel(" ", size=12)
plt.xlabel("Coffeeshop Reviews Sentiment ", size=12)
plt.show()

#################################################
## 5. Preparation of the Working Script
#################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import digits
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
from wordcloud import WordCloud

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv(r'C:\Users\Ozgur\OneDrive\Desktop\Miuul\Project\raw_yelp_review_data.csv', encoding='utf-8', delimiter=',', skipinitialspace=True)

df.loc[df["coffee_shop_name"] == "Caffé Medici ", ["coffee_shop_name"]] = "Caffe Medici "
df.loc[df["coffee_shop_name"] == "Lola Savannah Coffee Downtown ", ["coffee_shop_name"]] = "Lola Savannah Coffee Lounge "
df.loc[df["coffee_shop_name"] == "Summer Moon Coffee Bar ", ["coffee_shop_name"]] = "Summermoon Coffee Bar "

for i in df.index :
    remove_digits = str.maketrans('', '', digits)
    df["full_review_text"][i] = df["full_review_text"][i].translate(remove_digits)
    df["full_review_text"][i] = df["full_review_text"][i].lstrip(df["full_review_text"][i][0:3])
    #stringlerde sondan başlayıp eleman silme
    df["star_rating"][i] = df["star_rating"][i].rstrip(df["star_rating"][i][3:])

df["star_rating_num"] = df["star_rating"].astype(float)

df['coffee_shop_name'] = df['coffee_shop_name'].str.strip()

df['full_review_text'] = df['full_review_text'].str.lower()

df_wordcloud = df.copy()

stop_words = ["check-in", "check-ins"] + stopwords.words("english")
df["full_review_text"] = df["coffee_shop_description"] = df["full_review_text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

df["coffee_shop_description"] = df["coffee_shop_description"].str.replace(r'[^\w\s]+', '')
df1 = df.copy()

df = df.groupby(["coffee_shop_name"]).agg({"coffee_shop_description": "sum"}).reset_index()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.ylabel("count")
        plt.title(numerical_col)
        plt.show(block=True)

def content_based_recommender(coffee_shop_name, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['coffee_shop_name'])
    # title'ın index'ini yakalama
    cafe_index = indices[coffee_shop_name]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[cafe_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    cafe_indices = similarity_scores.sort_values("score", ascending=False)[1:4].index
    return dataframe['coffee_shop_name'].iloc[cafe_indices]


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['coffee_shop_description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = calculate_cosine_sim(df)

df1.groupby(['star_rating']).sum().plot(kind='pie', y='star_rating_num', autopct='%1.0f%%')
plt.show()

vader = SentimentIntensityAnalyzer()

function = lambda title: vader.polarity_scores(title)['compound']
df1['compound'] = df1['coffee_shop_description'].apply(function)

allWords = ' '.join([twts for twts in df_wordcloud['full_review_text']])

wordCloud = WordCloud(colormap= "flare", background_color="white", contour_color ="yellow",
                      width=500, height=300, random_state=21, max_font_size=110).generate(allWords)

plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()


def getAnalysis(score):
 if score < 0:
    return 'Negative'
 elif score == 0:
    return 'Neutral'
 else:
    return 'Positive'

df1['coffee_shop_description'] = df1['compound'].apply(getAnalysis)

df1.head(5)


plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df1['coffee_shop_description'].value_counts().plot(kind = 'bar')
plt.show()


df1.coffee_shop_description.value_counts().plot(kind='pie', autopct='%1.0f%%',  fontsize=8, figsize=(9,6), colors=["purple", "yellow", "pink"])
plt.ylabel(" ", size=12)
plt.xlabel("Coffeeshop Reviews Sentiment ", size=12)
plt.show()

