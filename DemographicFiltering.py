import numpy as np
import pandas as pd
pd.set_option('display.max_columns',None)




credit_data=pd.read_csv('./TMDB Data/tmdb_5000_credits.csv')
movies_data=pd.read_csv('./TMDB Data/tmdb_5000_movies.csv')

credit_data.columns=['id','title','cast','crew']
df=movies_data.merge(credit_data,on='id')
m=df['vote_count'].quantile(0.9)
C=df['vote_average'].mean()

def movie_ratings(x,m=m,C=C):
    v=x['vote_count']
    R=x['vote_average']
    return (v*R + m*C)/(v+m)



#print(m)
q_movies=df.copy().loc[df['vote_count']>=m]
#print(q_movies.shape)
q_movies['score']=q_movies.apply(movie_ratings,axis=1)
#print(q_movies.columns)
q_movies=q_movies.sort_values('score',ascending=False)
print(q_movies[['title_x','vote_count','vote_average','score']].head(10))

