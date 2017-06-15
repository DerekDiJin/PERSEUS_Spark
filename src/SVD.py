'''
Created on May 12, 2017

@author: DiJin
'''
import sys
import os
import re
import math
import random
import numpy as np
import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import Row

from linalg import Vectors
from linalg.distributed import RowMatrix

import Utility as ut
    
class SVD:
    
    def __init__(self):
        
        self._descriotion = 'In SVD'
        
        
    def statistics_compute(self, mat, dim, computeU):
        
        return mat.computeSVD(dim, computeU)

    '''
    input format: (v1, v2, ..., vdim)
    '''
    def add_approx(self, v_rdd, dim, N):
    
        result_rdd = v_rdd.keys()
        values = v_rdd.values()
        
        for i in xrange(dim):
            curCol = values.map(lambda x: x[i])
            [centers, counts] = self.prox(curCol, N)
            
            result_rdd = result_rdd.join(curCol.map(lambda x: self.findIndex(x, )))
            
        return
    
    def findIndex(self, value, min_value, max_value, N, centers):
        if value == max_value:
            return centers[N-1]
        else:
            interval = max_value - min_value
            grid = interval / N
            index = int(math.floor( (value-min_value) / grid))
        
            return centers[index]
    
    def linkProx(self, v_tuple_rdd, index, N):
        curCol_rdd = v_tuple_rdd.map(lambda x: float(x[index]))
        [centers, counts] = self.prox(curCol_rdd, N)
        
        val_min = curCol_rdd.min()
        val_max = curCol_rdd.max()
        val_val_t_rdd = curCol_rdd.map(lambda x: (x, self.findIndex(x, val_min, val_max, N, centers))).distinct()
    
        return val_val_t_rdd
    
    
    # todo: smarter coding?
    def addApprox(self, sc, v_df, v_rdd, N):   
                
        sqlContext = SQLContext(sc)

        val_val_t_rdd = self. linkProx(v_rdd, 1, N)    
        curV_df = sqlContext.createDataFrame(val_val_t_rdd, ['v_1', 'v_1_t'])
        v_df = v_df.join(curV_df, v_df.v_1 == curV_df.v_1).drop(curV_df.v_1)
        
        val_val_t_rdd = self. linkProx(v_rdd, 2, N)    
        curV_df = sqlContext.createDataFrame(val_val_t_rdd, ['v_2', 'v_2_t'])
        v_df = v_df.join(curV_df, v_df.v_2 == curV_df.v_2).drop(curV_df.v_2)
        
        val_val_t_rdd = self. linkProx(v_rdd, 3, N)    
        curV_df = sqlContext.createDataFrame(val_val_t_rdd, ['v_3', 'v_3_t'])
        v_df = v_df.join(curV_df, v_df.v_3 == curV_df.v_3).drop(curV_df.v_3)
        
        val_val_t_rdd = self. linkProx(v_rdd, 4, N)    
        curV_df = sqlContext.createDataFrame(val_val_t_rdd, ['v_4', 'v_4_t'])
        v_df = v_df.join(curV_df, v_df.v_4 == curV_df.v_4).drop(curV_df.v_4)
        
        val_val_t_rdd = self. linkProx(v_rdd, 5, N)    
        curV_df = sqlContext.createDataFrame(val_val_t_rdd, ['v_5', 'v_5_t'])
        v_df = v_df.join(curV_df, v_df.v_5 == curV_df.v_5).drop(curV_df.v_5)
        
        val_val_t_rdd = self. linkProx(v_rdd, 6, N)    
        curV_df = sqlContext.createDataFrame(val_val_t_rdd, ['v_6', 'v_6_t'])
        v_df = v_df.join(curV_df, v_df.v_6 == curV_df.v_6).drop(curV_df.v_6)
        
        val_val_t_rdd = self. linkProx(v_rdd, 7, N)    
        curV_df = sqlContext.createDataFrame(val_val_t_rdd, ['v_7', 'v_7_t'])
        v_df = v_df.join(curV_df, v_df.v_7 == curV_df.v_7).drop(curV_df.v_7)
        
        val_val_t_rdd = self. linkProx(v_rdd, 8, N)    
        curV_df = sqlContext.createDataFrame(val_val_t_rdd, ['v_8', 'v_8_t'])
        v_df = v_df.join(curV_df, v_df.v_8 == curV_df.v_8).drop(curV_df.v_8)
        
        val_val_t_rdd = self. linkProx(v_rdd, 9, N)    
        curV_df = sqlContext.createDataFrame(val_val_t_rdd, ['v_9', 'v_9_t'])
        v_df = v_df.join(curV_df, v_df.v_9 == curV_df.v_9).drop(curV_df.v_9)
        
        val_val_t_rdd = self. linkProx(v_rdd, 10, N)    
        curV_df = sqlContext.createDataFrame(val_val_t_rdd, ['v_10', 'v_10_t'])
        v_df = v_df.join(curV_df, v_df.v_10 == curV_df.v_10).drop(curV_df.v_10)
        
        
        return v_df
    '''
    binCenter returns the center of bin given boundary of bin
    binBoundary: array([double])
    centers: array([double])
    '''
    def bin_center(self, binBoundary):
        length = len(binBoundary)
        centers = [0] * (length-1)
        for idx in range(len(centers)):
            centers[idx] = binBoundary[idx] + (binBoundary[idx+1] - binBoundary[idx])/2.0

        return centers

    def prox(self, curCol, binNum):

        histList = curCol.histogram(binNum)
        centers = self.bin_center(histList[0])
        return [centers, histList[1]]
            
            
            