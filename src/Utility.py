'''
Created on Feb 26, 2017

@author: DiJin
'''
import sys
import os
import re
import random
import numpy as np
import pyspark
from pyspark import SparkConf, SparkContext
    
# 
# parse the raw data, rendering it to start with index 0
def parse_raw_lines(line):
    if line.startswith('#'):
        return -1, -1
    else:
        line = line.strip('\r\n')
        parts = re.split('\t| ', line)
        x_id = int(parts[0])
        y_id = int(parts[1])
        return x_id, y_id

def flatmap_add_index(line):
    row_index = line[1]
    row = line[0]
    for col_index in range(len(row)):
        yield col_index, (row_index, row[col_index])

def map_make_col(line):
    col_index = line[0]
    col = line[1]
    res = []
    for item in sorted(col):
        print (item)
        res.append(item[1])
    return col_index, np.array(res)

#
# print all the elements in an RDD
def printRDD(input):
#     fOut = open (outputFilePath, 'aw')
    for ele in input.collect():
        print(ele)
#         fOut.write(str(ele) + '\n')
    print ("-------------------------")

# print all the elements in an RDD
def printRDDList(input):
#     fOut = open (outputFilePath, 'aw')
    temp = input.map(lambda x: (x[0], list(x[1]))).collect()
    for ele in temp:
        print(ele)
#         fOut.write(str(ele) + '\n')
    print ("-------------------------")

def toCSVLine(data):
  return ','.join(str(d) for d in data)

def toTSVLine(data):
  return '\t'.join(str(d) for d in data)


# update through SGD
def egoProperties_by_partition(iterator, broadcastD, broadcastTotal):
    node_set = set()
    node = 0
#     H_set = set()

    for ele in iterator:
        node = ele[0]
        (src, dst) = ele[1]
        node_set.add(src)
        node_set.add(dst)
        
        
    results = []
    
    temp = []
    for i in node_set:
        temp.append(i)
        
    
    srcs = [0] * broadcastTotal.value
    dsts = [0] * broadcastTotal.value
    all = [0] * broadcastTotal.value
#     
    for index, item in enumerate(broadcastD.value):

        if item[0] in temp:
            srcs[index] = 1
        if item[1] in temp:
            dsts[index] = 1
#     print srcs
#     print '-----------------------------'
#     print dsts

    for i in xrange(broadcastTotal.value):
        all[i] = srcs[i] + dsts[i]
    
    edgeCount = all.count(2)
        
    
    results.append(('ego_edge', node, edgeCount))  
    results.append(('ego_size', node, len(node_set)))

 
    return results

#########################################################################################

