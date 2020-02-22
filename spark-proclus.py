#!/usr/bin/env python
# coding: utf-8

# In[71]:


# All proclus functions are defined below

def manhattanSegmentalDist(x, y, Ds):
	""" Compute the Manhattan Segmental Distance between x and y considering
		the dimensions on Ds."""
	dist = 0
	for d in Ds:
		dist += np.abs(x[d] - y[d])
	return dist / len(Ds)

def assignPoints(X, y, rdd_data, Mcurr, Dis):
	#shp = X.count()
	assigns = np.ones(y.shape[0]) * -1
	#for i in range(y.shape[0]):
	minDist = np.ones(y.shape[0]) * 10000 #np.inf
	best = -1
	for j in range(len(Mcurr)):
		a = rdd_data[Mcurr[j]]
		#print(type(a))        
		dist = X.map(lambda x: manhattanSegmentalDist(x, a, Dis[j])).collect()#Find the manhattan distance of each points in the data
		#dist = dis.first()#dist = dist.astype(np.int32)
		#print("dis is:",dist,"itertion",j) # To print distance matrix and to compare them
		for i in range(y.shape[0]):
			if dist[i] < minDist[i]:
				minDist[i] = dist[i]
				best = Mcurr[j]
				assigns[i] = best
	#print(assigns)
	return assigns

def evaluateClusters(X, size, assigns, Dis, Mcurr):

	upperSum = 0.0
	clust = []
	for i in range(len(Mcurr)):
		for j in range(len(assigns)):
			if Mcurr[i] == assigns[j]:
				clust.append(j)
		C = X.filter(lambda x: any(x[0] == clust[k] for k in range(len(clust))))#.filter(lambda x: x[0] == np.where(assigns == Mcurr[i])[0])# points in cluster M_i
		#print(C.count()) # number of points in cluster
		c_size = C.count()  # number of points in cluster
		Cm = C.reduce(lambda x,y: np.add(x,y)) / c_size # cluster centroid
		#print(Cm) # print centroid
		Ysum = 0.0

		for d in Dis[i]:
			g = X.map(lambda x: np.abs(x[d] - Cm[d])).reduce(lambda x,y: np.add(x,y))
			Ysum += g / c_size # avg dist to centroid along dim d:
		wi = Ysum / len(Dis[i])
		#print("wi is :",wi) # weighted average of cluster
		upperSum += c_size * wi

	return upperSum / size

def computeBadMedoids(X, assigns, Dis, Mcurr, minDeviation):
	N = X.count()
	k = len(Mcurr)
	Mbad = []
	counts = [len(np.where(assigns == i)[0]) for i in Mcurr]
	cte = int(np.ceil((N / k) * minDeviation))

	# get the medoid with least points:
	Mbad.append(Mcurr[np.argsort(counts)[0]])

	for i in range(len(counts)):
		if counts[i] < cte and Mcurr[i] not in Mbad:
			Mbad.append(Mcurr[i])

	return Mbad

def proclus(X, y, k = 2, l = 3, minDeviation = 0.1, A = 30, B = 3, niters = 30, seed = 1234):
	""" Run PROCLUS on a database to obtain a set of clusters and 
		dimensions associated with each one.
		Parameters:
		----------
		- X: 	   		the data set
		- k: 	   		the desired number of clusters
		- l:	   		average number of dimensions per cluster
		- minDeviation: for selection of bad medoids
		- A: 	   		constant for initial set of medoids
		- B: 	   		a smaller constant than A for the final set of medoids
		- niters:  		maximum number of iterations for the second phase
		- seed:    		seed for the RNG
	"""
	np.random.seed(seed)

	N, d = y.shape

	if B > A:
		raise Exception("B has to be smaller than A.")

	if l < 2:
		raise Exception("l must be >=2.")

	###############################
	# 1.) Initialization phase
	###############################

	# first find a superset of the set of k medoids by random sampling
	idxs = np.arange(N)      #get ids to randomize
	rdd_data = X.collect()   #rdd data stored in rows

	np.random.shuffle(idxs)  #random shuffle of all id points 
	S = idxs[0:(A*k)]        #get subset of data points
	M = greedy(X, rdd_data, S, B * k)  #get medoids
	print("medoids are ",M)
	###############################
	# 2.) Iterative phase
	###############################

	BestObjective = np.inf
        

	size = X.count() # number of rows 
	Mcurr = np.random.permutation(M)[0:k] # M current
	Mbest = None # Best set of medoids found

	
	it = 0 # iteration counter
	L = [] # locality sets of the medoids, i.e., points within delta_i of m_i.
	Dis = [] # important dimensions for each cluster
	assigns = [] # cluster membership assignments

	while True:
		it += 1
		L = []#print(it)

		for i in range(len(Mcurr)):
			mi = Mcurr[i]
			val = np.setdiff1d(Mcurr, mi)
			#print(val)
			D, di = dist_near(X,rdd_data,val,mi)# compute delta_i, the distance to the nearest medoid of m_i:
			#print(di)# compute L_i, points in sphere centered at m_i with radius d_i
			L.append(np.where(D <= di)[0])

		#print(L)# find dimensions:
		Dis = findDimensions(X, y, rdd_data, k, l, L, Mcurr)
		print("dimensions are:",Dis)
		# form the clusters:
		assigns = assignPoints(X, y, rdd_data, Mcurr, Dis)
		print("Assigned points are:",assigns)
		print("Current M is:",Mcurr)

		#here you are going to create a function
		#def f(x):
		#	d = {}
		#	for i in range(len(x)):
		#		d[str(i)] = x[i]
		#	return d

		#Now populate that
		#df = X.map(lambda x: Row(**f(x))).toDF()
		#df = X.toDF()
		#df = spark.createDataFrame(X, df.schema.add("assigns", IntegerType))
		#print(df)       
		ObjectiveFunction = evaluateClusters(X, size, assigns, Dis, Mcurr)  # evaluate the clusters:
		#print(ObjectiveFunction)   # print the cost  function of our clusters
		badM = [] # bad medoids

		Mold = Mcurr.copy()

		if ObjectiveFunction < BestObjective:
			BestObjective = ObjectiveFunction
			Mbest = Mcurr.copy()
			# compute the bad medoids in Mbest:
			badM = computeBadMedoids(X, assigns, Dis, Mcurr, minDeviation)
			print ("bad medoids:")
			print (badM)

		if len(badM) > 0:
			# replace the bad medoids with random points from M:
			print ("old mcurr:")
			print (Mcurr)
			Mavail = np.setdiff1d(M, Mbest)
			newSel = np.random.choice(Mavail, size = len(badM), replace = False)
			Mcurr = np.setdiff1d(Mbest, badM)
			Mcurr = np.union1d(Mcurr, newSel)
			print ("new mcurr:")
			print (Mcurr)

		print ("finished iter: %d" % it)

		if np.allclose(Mold, Mcurr) or it >= niters:
			break

	print ("finished iterative phase...")

	###############################
	# 3.) Refinement phase
	###############################

	# compute a new L based on assignments:
	L = []
	for i in range(len(Mcurr)):
		mi = Mcurr[i]
		L.append(np.where(assigns == mi)[0])

	Dis = findDimensions(X, y, rdd_data, k, l, L, Mcurr)
	assigns = assignPoints(X, y, rdd_data, Mcurr, Dis)

	print("m curr is ",Mcurr)# handle outliers:
	
	deltais = np.zeros(k)
	#dist = X.map(lambda x: manhattanSegmentalDist(x, , Dis[j])).collect()  
	for i in range(k):
		minDist = np.inf
		for j in range(k):
			if j != i:
				a = rdd_data[Mcurr[i]] 
				b = rdd_data[Mcurr[j]]
				dist = manhattanSegmentalDist(a, b, Dis[i])
				if dist < minDist:
					minDist = dist
		deltais[i] = minDist

	# mark as outliers the points that are not within delta_i of any m_i:
	for i in range(len(assigns)):
		clustered = False
		for j in range(k):
			d = manhattanSegmentalDist(rdd_data[Mcurr[j]], rdd_data[i], Dis[j])
			if d <= deltais[j]:
				clustered = True
				break
		if not clustered:
			
			#print "marked an outlier"
			assigns[i] = -1

	return (Mcurr, Dis, assigns)

def computeBasicAccuracy(pred, expect):
	""" Computes the clustering accuracy by assigning
		a class to each cluster based on majority
		voting and then comparing with the expected
		class. """

	if len(pred) != len(expect):
		raise Exception("pred and expect must have the same length.")

	uclu = np.unique(pred)

	acc = 0.0

	for cl in uclu:
		points = np.where(pred == cl)[0]
		pclasses = expect[points]
		uclass = np.unique(pclasses)
		counts = [len(np.where(pclasses == u)[0]) for u in uclass]
		mcl = uclass[np.argmax(counts)]
		acc += np.sum(np.repeat(mcl, len(points)) == expect[points])

	acc /= len(pred)

	return acc


# In[44]:


def findDimensions(X, y, rdd_data, k, l, L, Mcurr):
	N, d = y.shape
	Dis = [] # dimensions picked for the clusters

	Zis = [] # Z for the remaining dimensions
	Rem = [] # remaining dimensions
	Mselidx = [] # id of the medoid indexing the dimensions in Zis and Rem

	for i in range(len(Mcurr)):
		mi = Mcurr[i]
		# Xij is the average distance from the points in L_i to m_i
		m = rdd_data[mi]
		Davg = X.filter(lambda x: any(x[0] == L[i][j] for j in range(len(L[i]))))# Xij is the average distance from the points in L_i to m_i
		#print("d avg count",Davg.count())
		Xi = Davg.map(lambda x: np.abs(x - m)).reduce(lambda x,y: np.add(x,y))#.reduce() # Xij here is an array, containing the avg dists in each dimension
		Xij = Xi / len(L[i])
		#print("XIJ IS :",Xij)
		Yi = Xij.sum() / d # average distance over all dimensions
		Di = [] # relevant dimensions for m_i
		si = np.sqrt(((Xij - Yi)**2).sum() / (d-1)) # standard deviations
		Zij = (Xij - Yi) / si # z-scores of distances

		o = np.argsort(Zij)# pick the smallest two:
		print(o)
		Di.append(o[0])
		Di.append(o[1])
		Dis.append(Di)

		for j in range(2,d):
			Zis.append(Zij[o[j]])
			Rem.append(o[j])
			Mselidx.append(i)

	if l != 2:
		# we need to pick the remaining dimensions

		o = np.argsort(Zis)
		
		nremaining = k * l - k * 2
		# print "still need to pick %d dimensions." % nremaining

		# we pick the remaining dimensions using a greedy strategy:
		j = 0
		while nremaining > 0:
			midx = Mselidx[o[j]]
			Dis[midx].append(Rem[o[j]])
			j += 1
			nremaining -= 1

	#print "selected:"
	#print Dis

	return Dis


# In[45]:


def greedy(X, rdd_data, S, k):
	print(X.first())# remember that k = B * k here...


	M = [np.random.permutation(S)[0]] # M = {m_1}, a random point in S

	A = np.setdiff1d(S, M) # A = S \ M
	#print (len(A))
	#print(len(A))
	dists = np.zeros(len(A))  # initialize distance vector

	#for i in range(len(A)):
	y = rdd_data[M[0]]   # need to get row point for the randomized index
	#print(y.features)#dis = X.filter
	dist = X.filter(lambda x: any(x[0] == A[j] for j in range(len(A))))# tranform rdd with only rows of indexes of randomised subset
	print(dist.count())
	dists = dist.map(lambda x: np.linalg.norm(x-y)).collect() # action on rdd to get euclidean distance
	print(dists)

	for i in range(1, k):
		# choose medoid m_i as the farthest from previous medoids

		midx = np.argmax(dists)
		mi = A[midx]
		
		M.append(mi)
				
		# update the distances, so they reflect the dist to the closest medoid:
		for j in range(len(A)):
			y = rdd_data[mi]
			b = rdd_data[A[j]]
			#print(b.features)#.map(lambda x: np.linalg.norm(X.features - y.features)).collect()
			dists[j] = min(dists[j], np.linalg.norm(b - y))  # find the closest one 
		
		# remove mi entries from A and dists:
		A = np.delete(A, midx)
		dists = np.delete(dists, midx)

	return np.array(M)


# In[77]:


def dist_near(X,data,val,mi):
    D = X.map(lambda x: np.linalg.norm(data[mi] - x)).collect()
    print("nearby dist:",D)
    mind = []
    for i in range(len(val)):
        mind.append(D[val[i]])
    di = min(mind)
    print("distance is:",di)
    return D, di


# In[61]:



#import packages
import pyspark
from numpy import genfromtxt
from pyspark.sql.types import Row
#preprocessing
from numpy import array
from math import sqrt
from numpy import genfromtxt
import pandas as pd
from pyspark.mllib.clustering import KMeans
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.types import *
from pyspark.sql import *
import numpy as np
from collections import Counter

#import arffreader as ar
import matplotlib.pyplot as plt
import sys
from pyspark.sql.functions import monotonically_increasing_id 
from pyspark.sql import SparkSession
from scipy.spatial.distance import pdist, squareform


# In[62]:


# creating id for each dataset set so that we can do operations on rdd
'''
data = pd.read_csv('/usr/lib/spark/examples/src/main/python/Major_proj/datasets/Archive/B-cell1.csv',delimiter=',')
l = data.shape[0]
idx = 0
new_col = np.arange(l)
data.insert(loc=idx, column='idx', value=new_col)
data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)
#print(data)
loc = "/usr/lib/spark/examples/src/main/python/Major_proj/datasets/bcell1.csv"
data.to_csv(loc, index=False)
'''


# In[63]:


#spark = SparkSession.builder.appName("proclus").getOrCreate()
#sc = spark.sparkContext
#sqlContext = SQLContext(sc)
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.appName('proclus').getOrCreate()

loc = "/usr/lib/spark/examples/src/main/python/Major_proj/datasets/bcell1.csv"

#FOR SPARK RDD
dat = sc.textFile(loc) # get dataset into rdd
header = dat.first() #extract header
dataRDD = dat.filter(lambda row : row != header)   #filter out header

csv_rdd = dataRDD.map(lambda row: row.split(","))  #split all the rows
print(csv_rdd.take(1))                             #print out first n rows
data_rdd = csv_rdd.map(lambda x: np.array(x, dtype=np.float32)) #convert the string type to float
data_rdd.cache()
print(type(data_rdd))
print(data_rdd.count())
print('potential # of columns: ', len(data_rdd.take(1)[0]))  
print(data_rdd.getNumPartitions())

#data_df = csv_rdd.toDF()
#df_index = data_df.select("*").withColumn("id", monotonically_increasing_id())
#data_rdd = df_index.rdd
#print(data_rdd)


# In[79]:


#import arffreader as ar

##### read data ##### various types of data reading is mentioned here.
# can use this to read different types of files
"""
#X = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0]).mapPartitions(readPointBatch).cache()
#y, sup = ar.readarff("/usr/lib/spark/examples/src/main/python/Major_proj/datasets/Archive/leukemia2.csv") #read from arff
#y = genfromtxt('/usr/lib/spark/examples/src/main/python/Major_proj/datasets/Archive/leukemia1.csv',delimiter=',',skip_header=1) #path to csv file
y = pd.read_csv('/usr/lib/spark/examples/src/main/python/Major_proj/datasets/Archive/leukemia1.csv',delimiter=',')
# change label values here
#print(y)
print(y.dtypes)
#df =  df.withColumn("cls", df["CLASS"].getItem(0).cast("string")) 
y = y.replace({'ALL': 0, 'AML': 1})
sup = [item[-1] for item in y]
sup = np.array(sup)
sup = sup.ravel()
print(y)
"""
g = pd.read_csv(loc,delimiter=',')
#print(y)
sup = g.iloc[:, -1]
print(sup)
y = g.iloc[:, :-1]
Dims = [0,1]
#plotter.plotDataset(X, D = Dims) # plot 0-1 dimensions

R = 1 # toggle run proclus
RS = 0 # toggle use random seed

if R: # run proclus
	rseed = 90288
	if RS:
		rseed = np.random.randint(low = 0, high = 1239831)

	print ("Using seed %d" % rseed)  #randomize seed
	k = 2    #initialize  no of clusters
	l = 1000    #get no of dimensions 
	M, D, A = proclus(data_rdd, y, k, l, seed = rseed) #running proclus code
	print("Final assigns are: ",A)
	e = Counter(A)
	print("Count in each cluster: ",e)
	#print "Accuracy: %.4f" % computeBasicAccuracy(A, sup)
	#print "Adjusted rand index: %.4f" % adjrand.computeAdjustedRandIndex(A, sup)

	#plotter.plotClustering(X, M, A, D = Dims)


# In[8]:



from collections import Counter

label = np.array(sup)
print(label)
predicts = A.astype(int)
print(predicts)
print(len(label),len(predicts))
#need to remove outliers
outlier = np.where(predicts==-1)[0]
print(outlier)
le = len(outlier)


print("removing outliers")
for i in range(le):
    label = np.delete(label, i)
    predicts = np.delete(predicts, i)
    #print (i)
print('After removing outliers:    ')
print(label)
print(predicts)

# need to index them differently to find f1 and purity
z = list(set(predicts))
print(z)
for i in range(k):
    a = z[i]
    #predicts[predicts == a] = i
    predicts = np.where(predicts == a, i, predicts)
    #np.place(predicts, predicts = a, [44, 55])
#print(predicts)

def max_ele(label,predicts,k):
    a = Counter(label) #label's count
    s = len(predicts)
    
    x = len(a)
    #print(s)
    ocr = []
    for i in range(x):
        temp = []
        for j in range(s):
            if label[j] == i:
                temp.append(predicts[j])
        ocr.append(temp)
    #print(ocr)        
    clus = []
    for i in range(x):
        y =  Counter(ocr[i])
        #print(y)
        clus_ocr = []
        for j in range(k):
            clus_ocr.append(y[j])
            #print(y[j])
        clus.append(clus_ocr)
    print(clus)
    
    maxi = []
    maxi_count =[]
    idx = 0
    for j in range(k):
        ma = 0 
        for i in range(x):
            if (ma < clus[i][j]):
                idx = i
            ma = max(clus[i][j],ma)
        maxi.append(idx)
        maxi_count.append(ma)
    #print(maxi)
    return maxi,clus


maxi,clus = max_ele(label,predicts,k) # best k value ,here k = 9
print(maxi)
print(clus)


# In[5]:


def metrics(maxi,label,clus,k):
    prec = []
    rec = []
    f1 = []
    ct = Counter(label)
    #print(ct)
    for j in range(k):
        x = (clus[maxi[j]][j])
        y = (sum([item[j] for item in clus]))
        z = float(x)/float(y)
        #print(z)
        prec.append(z)
        rc = (clus[maxi[j]][j])
        rec.append(float(rc)/ct[maxi[j]])
        f = (2*prec[j]*rec[j])/(prec[j]+rec[j])
        f1.append(f)

    print("precision : ",prec)
    print("recall are: ",rec)
    print("f1-score is: ",f1)
    return prec,rec,f1


precision,recall,f1_score = metrics(maxi,label,clus,k) # get metrics

f1_value = np.average(f1_score)

print("F1-Value of the clusters is: ",f1_value)


# In[6]:


def purity_fn(df,clus,maxi,k):
    shape = df.shape
    r_len = shape[0]
    num = 0
    for i in range(k):
        num = num + float(clus[maxi[i]][i])
    purity = num/r_len
    return purity
#purity = []
#for i in range(2,10):
purity = purity_fn(g,clus,maxi,k)
print("Purity of the clusters is:",purity)


# In[ ]:




