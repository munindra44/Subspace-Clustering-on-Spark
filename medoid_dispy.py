'''
Algorithm : Distributed Medoid
Author : Sandhya Harikumar
'''

#!/usr/bin/env python

from __future__ import division #for getting float value while dividing 2 integers
import sys
import time
import random
import math
import gc
import dispy
import socket
import os
import collections
from collections import defaultdict,Counter

def read_input(file):
    temp = []
    for line in file:
        line = line.replace('\n','')
        temp.append(line)
    return temp

def distance(p1,p2):
    sum = 0
    p1 = p1.split(',')
    p2 = p2.split(',')
    for i in range(1,len(p1)):  # 0'th position is medoid id
        sum += abs(float(p1[i]) - float(p2[i]))
    #print(sum)
    return sum

def coordinate_medoid(data,k):
    
    medoid = []    
    tmp = 1
    k = 1*k
    j = len(data)-1
    maxdiff = 0
    medoid_cnt = 1
    medcnt = []

    med = [0]*(j+1)
    premed = [0]*(j+1)
    
    medoid.append(random.choice(data))
    current_medoid = medoid[0]
    print ("current medoid ",data)
    while(tmp <= k):
        for i in range(0,j+1):
            med[i] = distance( current_medoid , data[i] )
            if tmp == 1:
                premed[i] = med[i]
                if maxdiff < med[i]:
                    maxdiff = med[i]    
            
            else:
                if premed[i] > med[i]:
                    premed[i] = med[i]
                    if maxdiff < med[i]:
                        maxdiff = med[i]
                else:
                    if maxdiff < premed[i] :
                        maxdiff = premed[i];

        i = 0
        idcnt = 0
        if tmp < k :
            i = 0
            idcnt = 0
            while(i <= j):    
                if premed[i] == maxdiff :
                    if idcnt == 0:
                        current_medoid = data[i]                        
                    premed[i] = 0
                    idcnt += 1
                i += 1
            #end while loop
            if maxdiff != 0:
                medoid.append(current_medoid);
            maxdiff = 0
            i = 0
            medoid_cnt =+ 1
       
        tmp += 1
    #end main while loop
    print("exiting greedy")
    return medoid

def greedy(k):
    print("hello")
    #home = os.path.expanduser("~")
  
    l1=[]
    #l1.append(home);
    l1.append("/usr/lib/spark/examples/src/main/python/Major_proj/datasets/diabetes1.csv");
    urlPath= "".join(l1);
    print(urlPath)
    input_file = open(urlPath,"r")  
    data = read_input(input_file)   

    medoid = []    
    tmp = 1
    k = 1*k
    j = len(data)-1
    maxdiff = 0
    medoid_cnt = 1
    medcnt = []

    med = [0]*(j+1)
    premed = [0]*(j+1)
    
    medoid.append(random.choice(data))
    current_medoid = medoid[0]
    print ("current medoid greedy "+current_medoid)
    while(tmp <= k):
        for i in range(0,j+1):
            med[i] = distance( current_medoid , data[i] )
            
            if tmp == 1:
                premed[i] = med[i]
                if maxdiff < med[i]:
                    maxdiff = med[i]    
            
            else:
                if premed[i] > med[i]:
                    premed[i] = med[i]
                    if maxdiff < med[i]:
                        maxdiff = med[i]
                else:
                    if maxdiff < premed[i] :
                        maxdiff = premed[i];

        i = 0
        idcnt = 0
        if tmp < k :
            i = 0
            idcnt = 0
            while(i <= j):    
                if premed[i] == maxdiff :
                    if idcnt == 0:
                        current_medoid = data[i]                        
                    premed[i] = 0
                    idcnt += 1
                i += 1
            #end while loop
            if maxdiff != 0:
                medoid.append(current_medoid);
            maxdiff = 0
            i = 0
            medoid_cnt =+ 1
       
        tmp += 1
    #end main while loop
    
    return medoid
 
def getstring(l):
	s = ''
	c = 0
	for i in range(len(l)):
		c = c+1
		s += l[i]
		if(c != len(l)):
			s += ","
	return s




'''

def nbhd(mrval):

    import pandas
    import collections
    from collections import defaultdict,Counter

    #home = os.path.expanduser("~")
  
    l1=[]
    #l1.append(home);
    l1.append("/usr/lib/spark/examples/src/main/python/Major_proj/datasets/");
    urlPath= "".join(l1);
    
    input_file = open(urlPath,"r")  
    rval = read_input(input_file)

    index = 0
    count = 0
    nearstmid = [0]*len(mrval)
    premed = [0]*len(mrval)
    med = [0]*len(mrval)
    dmed = [0]*len(rval)
    locality = []
 
    mnj = len(rval)-1
    j = len(mrval)-1

    while(count <= j):
        temp = mrval[index].split(',')        
        for i in range(0,j+1): #submedoid table
            temp1 = mrval[i].split(',')            
            if(count == 0 and i == 0):
                nearstmid.append(temp[0])
                premed[i] =9999999
            if(temp1[0] != temp[0]):
                med[i] = distance(mrval[index] , mrval[i])
                if(count == 0 and i != 0):
                    nearstmid[i] = temp[0]
                    premed[i] = med[i]
                else:
                    if(premed[i] > med[i]):
                        premed[i] = med[i]
                        nearstmid[i] = temp[0]
                 
         #end for loop          
        count += 1
        index += 1    
    #end of while loop
    index = 0
    while(index <= j):
        i = 0
        temp = mrval[index].split(',')        
        #delete from medoid table
        for i in range(0,mnj+1):
            dmed[i] = distance(mrval[index] , rval[i])
            if(dmed[i] <= premed[index]):
                temp1 = rval[i].split(',')
                temp1.insert( 0, temp[0])
                locality.append(getstring(temp1))                
         
        #end for loop
        index += 1
    #end while
    #An example of locality list ['4,461,84,50,97,42,62\n']. here 4 is medoid id, 461 is the sample id, remaining are the features.
    
    locl_matrix=[]
    for j in range(0,len(locality)-1):
    	loc_matrix =[int(i) for i in locality[j].split(',')]
    	locl_matrix.append(loc_matrix)
    #return locl_matrix

    #cntr=dict()
    #cntr=count_occurrences(locl_matrix)
    #return cntr

    flattened = [val[0] for val in locl_matrix]
    cntr = Counter(flattened)

    pd = pandas.DataFrame(locl_matrix)

    pd.to_csv('diabetes1.csv')

    
    return cntr
    
'''
def PROCLUS(data , K , Avgdim):

    
    Medoid = greedy(data,K)
    print(Medoid)
   

def main():
    #Stage 1
    # distribute 'greedy' to nodes; 'greedy' has dependencies (needed from client)
    cluster1 = dispy.JobCluster(greedy,nodes=['master'],depends=[distance,read_input])    
    cluster2 = dispy.JobCluster(greedy,nodes=['15cpu0110L'],depends=[distance,read_input])    
    jobs=[]
    Medoid=[]

    
    for i in range(2):
        job1=cluster1.submit(2)
        job1.id=i
        job2=cluster2.submit(2)
        job2.id=i
        jobs.append(job1)
        jobs.append(job2)

    for job in jobs:
        c_Medoid=job()
        host = socket.gethostname()
        print('%s executed job %s at %s' % (host, job.id, job.start_time))
        print(c_Medoid)
        Medoid.append(c_Medoid)
	
    #print(Medoid)  
    Medoid=coordinate_medoid(Medoid,3)
    print('Coordinate medoid:',Medoid)
     
    cluster1.print_status() 
    cluster1.close()
    cluster2.print_status()
    cluster2.close() 
    #End of Stage 1

 

    #End of Stage 2

    #Stage 3
#    cluster = dispy.JobCluster(nbhd,depends=[distance,read_input,getstring])        
 #   jobs            Fdimsn = findimension(K , AvgDim , Locality , noOfAttributes , subMedoid)
    #End of Stage 3
    
if __name__ == "__main__":
    main()
