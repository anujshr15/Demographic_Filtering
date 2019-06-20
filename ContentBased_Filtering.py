import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_columns',None)




credit_data=pd.read_csv('./TMDB Data/tmdb_5000_credits.csv')
movies_data=pd.read_csv('./TMDB Data/tmdb_5000_movies.csv')

credit_data.columns=['id','title','cast','crew']
df=movies_data.merge(credit_data,on='id')

tfidf=TfidfVectorizer(stop_words='english')

df['overview'].fillna('',inplace=True)
tfidf_matrix=tfidf.fit_transform(df['overview'])
#print(df.index)
cosine_sim=linear_kernel(tfidf_matrix,tfidf_matrix)

indices=pd.Series(df.index,index=df.title_x).drop_duplicates()

def get_recommendations(title,cosine_sim=cosine_sim):
    idx=indices[title]
    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    sim_scores=sim_scores[1:11]
    movie_indices=[i[0] for i in sim_scores]
    return df['title_x'].iloc[movie_indices]

features=['cast','crew','keywords','genres']
for feature in features:
    df[feature]=df[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if(i['job']=='Director'):
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x,list):
        names=[i['name'] for i in x]
        if len(names)>3:
            names=names[:3]
        return names
    return []

df['director']=df['crew'].apply(get_director)

features=['cast','keywords','genres']
for feature in features:
    df[feature]=df[feature].apply(get_list)

#print(df[['title_x','cast','director','keywords','genres']].head(3))

def clean_data(x):
    if isinstance(x,list):
        return [str.lower(i.replace(" ","")) for i in x]
    else:
        if(isinstance(x,str)):
            return str.lower(x.replace(" ",""))
        return ""

features=['cast','keywords','director','genres']
for feature in features:
    df[feature]=df[feature].apply(clean_data)


def create_soup(x):
    return " ".join(x['keywords'])+" "+" ".join(x['cast'])+" "+" ".join(x['director'])+" "+" ".join(x['genres'])

df['soup']=df.apply(create_soup,axis=1)

count=CountVectorizer(stop_words='english')
count_matrix=count.fit_transform(df['soup'])

cosine_sim2=cosine_similarity(count_matrix,count_matrix)

df=df.reset_index()

indices=pd.Series(df.index,index=df['title_x'])

print(get_recommendations('The Godfather',cosine_sim2))


