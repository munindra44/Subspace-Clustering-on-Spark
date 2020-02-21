
# coding: utf-8

# In[1]:


import numpy as np
from numpy import genfromtxt
import pandas as pd

df = pd.read_csv('/home/munindra/Desktop/Final_Yr_Proj/diabetes.csv',delimiter=',') #path to csv file
df.head()


# In[2]:


# separate label from file
label = df.as_matrix(columns=df.columns[9:])
label = label.ravel()

df = df.drop(['Outcome'],axis = 1)
print(label)


# In[3]:


# remove id column
df = df.drop(['id'],axis = 1)
df.head()


# In[22]:


#k means clustering
from sklearn.cluster import KMeans
from time import time
import matplotlib.pyplot as plt

cost = np.zeros(40)
for i in range(2,40):
    start = time()
    kmeans = KMeans(n_clusters=i)
    model = kmeans.fit(df)
    labels = model.labels_
    cost[i] = model.inertia_
    print ("k:",i, " cost:", cost[i])
    end = time()
    print ("K means from sklearn took {:.4f} seconds(k = {:.4f})".format(end - start,i))


# In[23]:


#plot elbow curve
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,40),cost[2:40])
ax.set_xlabel('k')
ax.set_ylabel('cost')


# In[30]:


# k = 9 looks better value
best_k = 23 # change value of k here
kmeans = KMeans(n_clusters=best_k, random_state=0).fit(df)
centr = kmeans.cluster_centers_
predicts = kmeans.predict(df)


# In[31]:


print(centr)


# In[32]:


#print(predicts)
#print(label)


# In[33]:


def max_ele(label,predicts,k):
    a = np.zeros(k) #0's count
    s = len(predicts)
    b = np.zeros(k) #1's count
    for i in range(s):
        temp = predicts[i]
        if label[i] == 1:
            b[temp] = b[temp] + 1 
        else:
            a[temp] = a[temp] + 1

    
    maxi = np.zeros(k)
    for i in range(k):
        #print(a[i])
        #print(b[i])
        if a[i]<b[i]:
            maxi[i] = 1
        else:
            maxi[i] = 0
    return maxi,a,b

maxi,a,b = max_ele(label,predicts,best_k) # best k value ,here k = 9
print(a,b) # a = total number of 0 labelled data , b = 1 labelled data
print(maxi) # maxi has most labelled data in a cluster ( either 0 or 1)


# In[34]:


from collections import Counter

def metrics(a,b,maxi,label,k):
    prec = np.zeros(k)
    rec = np.zeros(k)
    f1 = np.zeros(k)
    z = Counter(label)
    #print(z[0])
    for i in range(len(maxi)):
        if maxi[i]==0:
            prec[i] = a[i]/(a[i]+b[i])
            rec[i] = a[i]/z[0]
        else:
            prec[i] = b[i]/(a[i]+b[i])
            rec[i] = b[i]/z[1]
        f1[i] = (2*prec[i]*rec[i])/(prec[i]+rec[i])
    #print("precision are:",prec)
    #print("recall are:",rec)
    print("f1-score is:",f1)
    return prec,rec,f1


precision,recall,f1_score = metrics(a,b,maxi,label,best_k) # get metrics

f1_value = np.average(f1_score)

print("F1-Value of the clusters is: ",f1_value)


# In[35]:



def purity(df,a,b,k):
    shape = df.shape
    r_len = shape[0]
    num = 0
    for i in range(k):
        if a[i]>b[i]:
            num = num + a[i]
        else:
            num = num + b[i]
    purity = num/r_len
    return purity

purity = purity(df,a,b,best_k)
print(purity)

