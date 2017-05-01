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

import Utility as ut
    
class Degrees:
    
    def __init__(self):
        self._descriotion = 'In degrees'
        
    def statistics_compute(self, D, mod):
        
        if mod == 'out':
            output_rdd = D.groupByKey().map(lambda r: (r[0], len(r[1])))
            print("out degree completed")
            return output_rdd
        
        if mod == 'in':
            output_rdd = D.map(lambda r: (r[1], r[0])).groupByKey().map(lambda r: (r[0], len(r[1])))
            print("in degree completed")
            return output_rdd
        
        if mod == 'total':
            output_rdd = D.union(D.map(lambda r: (r[1], r[0]))).groupByKey().map(lambda r: (r[0], len(r[1])))
            print("total degree completed")
            return output_rdd
        
        if mod == 'weighted_out':
            output_rdd = D.map(lambda x: (x[0], x[2])).groupByKey().map(lambda r: (r[0], sum(r[1])))
            print("out weighted degree completed")
            return output_rdd

        if mod == 'weighted_in':
            output_rdd = D.map(lambda x: (x[1], x[2])).groupByKey().map(lambda r: (r[0], sum(r[1])))
            print("in weighted degree completed")
            return output_rdd
            
        if mod == 'weighted_total':
            output_rdd = D.map(lambda x: (x[0], x[2])).union(D.map(lambda x: (x[1], x[2]))).groupByKey().map(lambda r: (r[0], sum(r[1])))
            print("total weighted degree completed")
#             ut.printRDD(output_rdd)
            return output_rdd


    def deg_vs_count(self, total_degree_rdd):
        output_rdd = total_degree_rdd.map(lambda x:(x[1], x[0])).groupByKey().map(lambda x: (x[0], len(x[1])))
        return output_rdd
    
    
    '''
    binCenter returns the center of bin given boundary of bin
    binBoundary: array([double])
    centers: array([double])
    '''
    def bin_center(self,binBoundary):
        length = len(binBoundary)
        centers = [0] * (length-1)
        for idx in range(len(centers)):
            centers[idx] = binBoundary[idx] + (binBoundary[idx+1] - binBoundary[idx])/2.0

        return centers

    def deg_vs_count_weight(self, totalDegree, binNum):
        deg = totalDegree.map(lambda x:x[1])
        histList = deg.histogram(binNum)
        centers = self.bin_center(histList[0])
        return [centers, histList[1]]
            
            
            