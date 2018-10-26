
# coding: utf-8

# In[ ]:


sc


# In[ ]:


#restore session notebook
#import dill
#dill.load_session("notebook_env.db")


# In[ ]:


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.types import *


# In[ ]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import time


# In[ ]:


#lines = spark.read.csv("gs://dataset-rs/dataset-last.fm/dataset_lastfm.csv", header="true",inferSchema="true").rdd
lines = spark.read.csv("/home/aleja/Documentos/datasets/meu-lastfm/dataset_lastfm_v3_ConvertIds.csv", header="true",inferSchema="true").rdd

lines.take(2)


# In[ ]:


#Create dataframe
ratings = spark.createDataFrame(lines) 
#ratings = ratings.withColumn("topTracks_playcount", col("topTracks_playcount").cast('int'))

#50% Sample of data 
df_ratings = ratings.sample(False,fraction=0.5, seed=1)
#Drop NaN data 
df_ratings = df_ratings.na.drop()
df_ratings.select('new_userId').count()


# In[ ]:


df_ratings.show(5)


# In[ ]:


#changes the names of rows
df_ratings =  df_ratings.selectExpr("new_userId as userId","name as name", "new_songId as songId","track_name as track_name","new_artistId as artistId", "artist_name as artist_name","track_playcount as track_playcount")


# In[ ]:


# transform the type of data
df_ratings = df_ratings.withColumn("userId", col("userId").cast("int"))
df_ratings = df_ratings.withColumn("songId", col("songId").cast("int"))
df_ratings = df_ratings.withColumn("artistId", col("artistId").cast("int"))
df_ratings = df_ratings.withColumn("track_playcount", col("track_playcount").cast("int"))


# In[ ]:


## How many distinct users in data ?

uniqueUsers = df_ratings.select('userId').distinct().count()
print("Total n. of users: ", uniqueUsers)

## How many distinct artists in data ?

uniqueArtists  = df_ratings.select("artistId").distinct().count()
print("Total n. of artists: ", uniqueArtists)

## How many distinct music in data ?

uniqueSongs  = df_ratings.select("songId").distinct().count()
print("Total n. of songs: ", uniqueSongs)


# In[ ]:


#select users to play a song more than 10 times and less to 300 
raw_plays_df_2more_plays = df_ratings.filter(df_ratings.track_playcount >= 10).distinct()

tot_entries_2more = raw_plays_df_2more_plays.count()
print('Total enties with two or more plays: {0}'.format(tot_entries_2more))

raw_plays_df_2more_plays = raw_plays_df_2more_plays.filter(raw_plays_df_2more_plays.userId < (uniqueUsers))                                                    .select('userId',"name", 'songId',"track_name",'artistId',"artist_name", 'track_playcount').orderBy('track_playcount',ascending=False) 
raw_plays_df_2more_plays.cache() 


# In[ ]:



df_ratings_less100 = raw_plays_df_2more_plays.filter(raw_plays_df_2more_plays.track_playcount <= 300).distinct()
df_ratings_less100.count()


# ### create music data

# In[ ]:


music_data =  df_ratings_less100.selectExpr("songId","track_name","artistId","artist_name")
music_data


# In[ ]:


sub_rating_data = df_ratings_less100.select("userId","songId","track_playcount")
sub_rating_data = sub_rating_data.na.drop()
sub_rating_data.orderBy(col('userId'),col('songId')).show(10)


# ### Train and test data

# In[ ]:


(training, test) = sub_rating_data.randomSplit([0.8, 0.2])
als = ALS(rank=15, maxIter=10, regParam=0.01, alpha=0.5, implicitPrefs=True, userCol="userId", itemCol="songId", ratingCol="track_playcount", coldStartStrategy="drop")
model = als.fit(training)


# In[ ]:


##Predictions-  test model
predictions = model.transform(test)
predictions.printSchema()
predictions.show()


# In[ ]:


#p1 =  predictions
#t = p1.groupBy("new_userId","new_songId").count()
#t.orderBy("new_userId","count", ascending=False).show()



# In[ ]:


evaluator = RegressionEvaluator(labelCol="track_playcount", predictionCol="prediction")
rmse = evaluator.evaluate(predictions.na.drop(), {evaluator.metricName :"rmse"})
mae =  evaluator.evaluate(predictions.na.drop(), {evaluator.metricName :"mae"})
print("Root-mean-square error = " + str(rmse))
print("mae = " + str(mae))



# ## Función para recomendación

# ### Recomendação personalizada para um Usuario

# In[ ]:


from pyspark.sql.functions import lit



def recommendMusic(model, user, nbRecommendations):
     # Create a Spark DataFrame with the specified user and all the songs ratings in DataFrame
    dataSet = df_ratings_less100.select("songId").distinct().withColumn("userId", lit(user))

    # Create a Spark DataFrame with the movies that have already been rated by this user
    musicAlreadyRated = df_ratings_less100.filter(df_ratings_less100.userId == user).select("songId", "userId")
    #subtrama
    #sub_musicAlreadyRated =  musicAlreadyRated.sample(False,fraction=0.5, seed=1)
    #musicInclude = musicAlreadyRated.subtract(sub_musicAlreadyRated)
    #musicInclude_2 = musicInclude.join(music_data, musicInclude.new_songId == music_data.new_songId).distinct().select(musicInclude.new_songId, music_data.topTracks_name, music_data.topTracks_artist_name)
    #print ("music rated that was include to predictions:")
    #musicInclude_2.show(50)
    # Apply the recommender system to the data set without the already rated movies to predict ratings
    predictions = model.transform(dataSet.subtract(musicAlreadyRated)).dropna().select("songId", "prediction").orderBy("prediction", ascending=False).limit(nbRecommendations)
    
    # Join with the ratings DataFrame to get the music titles and genres
    recommendations = predictions.join(music_data, predictions.songId == music_data.songId).distinct().select(predictions.songId, music_data.track_name, music_data.artist_name, predictions.prediction).orderBy("prediction", ascending=False)
    recommendations.show(truncate=False)


# ### musicas escutadas pelo usuario

# In[ ]:


### songs listened - user
def songs_listened(userId, Nsongs):
    musicAlreadyRated = df_ratings_less100.filter(df_ratings_less100.userId == userId).select("songId", "userId","track_playcount").distinct()
    music_listened =  musicAlreadyRated.join(music_data, musicAlreadyRated.songId == music_data.songId).select(musicAlreadyRated.songId, music_data.track_name, music_data.artist_name,musicAlreadyRated.track_playcount).distinct()
    music_listened.orderBy("track_playcount",ascending=False).show(Nsongs)


# ### Dado um item, probabilidade que um Usuario goste 

# In[ ]:


def rankUserforItem(model, itemID, friendID):
    x = df_ratings_less100.select("userId").distinct().withColumn("songId", lit(itemID))
    #print ("Os DADOS {}" .format(x.show(truncate=False)))
    
    predictions = model.transform(x).dropna().select("userId","songId", "prediction").orderBy("prediction", ascending=False)
    
    recommendations =  predictions.join(music_data, predictions.songId ==  music_data.songId).distinct().select(predictions.userId, predictions.songId, music_data.track_name, music_data.artist_name, predictions.prediction).orderBy("prediction", ascending=False)
    rankUser = recommendations.filter(recommendations.userId==friendID)
    
    rankUser.show()
    
    
    
    


# In[ ]:


u=191
print ("list of songs listened by user {}:" .format(u))

songs_listened(u,500)


# In[ ]:


u = 191
print ("Recommendations for user {}:".format(u))
recommendMusic(model,u,30)


# In[ ]:


i= 90242
f=191

print("Rank of User= {} for item= {}" .format(f,i))
rankUserforItem(model,i,f) 


# In[ ]:


u = 14600
print ("list of songs listened by user {}:" .format(u))

songs_listened(u,1000)


# In[ ]:


print ("Recommendations for user {}:".format(u))
recommendMusic(model,u,20)


# In[ ]:


#save notebook session 
#import dill
#dill.dump_session("notebook_env.db")

