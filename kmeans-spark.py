
# coding: utf-8

# In[1]:


from pyspark import SparkContext
import pyspark


# In[2]:


from pyspark import SparkContext
sc = SparkContext.getOrCreate()


# In[17]:


from numpy import array
from math import sqrt
from numpy import genfromtxt
import pandas as pd
from pyspark.mllib.clustering import KMeans

sqlContext = SQLContext(sc)

#Load and parse the data
dataset = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('/home/munindra/Desktop/Final_Yr_Proj/diabetes.csv')
df = pd.read_csv('/home/munindra/Desktop/Final_Yr_Proj/diabetes.csv',delimiter=',') # path to file here

label = df.as_matrix(columns=df.columns[9:])
label = label.ravel()
dataset = dataset.na.drop()

dataset.show()


# In[25]:


dft = dataset.select(*(dataset[c].cast("float").alias(c) for c in dataset.columns[1:]))

from pyspark.sql.functions import monotonically_increasing_id 

dft = dft.select("*").withColumn("id", monotonically_increasing_id())

#split outcome
label = dft[["Outcome"]]
dft = dft.drop('Outcome')
dft.head()


# In[26]:


from pyspark.ml.feature import VectorAssembler # in spark api we have to convert data to vectors in order to run the model.
# kmeans is imported from pyspark.ml.clustering
from pyspark.ml.clustering import KMeans

# creating an instance of the vector assembler 
assembler = VectorAssembler(inputCols = dft.columns, outputCol = 'features')

# transforming dataframe into vector assembler 
final_df = assembler.transform(dft)
final_df.drop('Outcome')

final_df.show(5)


# In[27]:


# in clustering we have to scale the features so as to reduce the distance and which helps in computation become faster
from pyspark.ml.feature import StandardScaler

# created instance of the standard scaler
scaler = StandardScaler(inputCol = 'features', outputCol = 'scaledFeatures')

#fitting the vector data and transforming with scaler transformation
scaler_model = scaler.fit(final_df)
final_df = scaler_model.transform(final_df)
final_df.show(6)


# In[28]:


import numpy as np
import matplotlib.pyplot as plt
from time import time


cost = np.zeros(20)
for k in range(2,20):
    start = time()
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(final_df.sample(False,0.1, seed=42))
    cost[k] = model.computeCost(final_df)
    end = time()
    print ("K means from spark took {:.4f} seconds(k = {:.4f})".format(end - start,k))


# In[8]:


fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),cost[2:20])
ax.set_xlabel('k')
ax.set_ylabel('cost')


# In[39]:


best_k = 15 # choose best k from elbow curve
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(final_df)
centers = model.clusterCenters()

print("Cluster Centers: ")
for center in centers:
    print(center)


# In[40]:


predicts = model.transform(final_df)


# In[1]:


#predicts.show(40)
#df.select("prediction").collect()

