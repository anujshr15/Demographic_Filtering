import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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

print(get_recommendations('The Dark Knight'))
