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
from operator import add
from pyspark import SparkConf, SparkContext

import Utility as ut

class PageRank:
    
    def __init__(self):
        self._descriotion = 'In pagerank'

        
    def computeContribs(self, urls, rank):
        """Calculates URL contributions to the rank of other URLs."""
        
        if urls is not None:
            num_urls = len(urls)
            for url in urls:
                yield (url, rank / num_urls)

                
    def computeContribs_weighted(self, urls_w, rank):
        """Calculates URL contributions to the rank of other URLs."""

        if urls_w is not None:
            num_urls = len(urls_w)
            temp_sum = 0
            temp  = [(ele[0], ele[1] / num_urls) for ele in urls_w]
            for ele in temp:
                temp_sum = temp_sum + ele[1]

            for url in temp:
                yield (url[0], url[1] * rank / temp_sum)
        
    def statistics_compute(self, D, Iter, d, debug_mod):
        max_src, max_dst = D.reduce(lambda x, y: (max(x[0], y[0]), max(x[1], y[1])))
        print(max_src, max_dst)
        
        N = float(max(max_src, max_dst))
        src_nodes = D.map(lambda x: (x[0], 0)).distinct()
        dst_nodes = D.map(lambda x: (x[1], 0)).distinct()
        src_only = src_nodes.subtractByKey(dst_nodes).cache()
        
        ranks = D.map(lambda line: line[0]).union(D.map(lambda line: line[1])).distinct().map(lambda line: (line, 1/N)).cache()
        
        for iter in xrange(Iter):
            if debug_mod == 1:
                print(iter)
                ut.printRDD(D)
            contribs = D.groupByKey().join(ranks).flatMap( lambda url_urls_rank: self.computeContribs(url_urls_rank[1][0], url_urls_rank[1][1]) )
            contribs = contribs.union(src_only)

            if debug_mod == 1:
                ut.printRDD(contribs)

            ranks = contribs.reduceByKey(add).mapValues(lambda rank: rank * d + (1-d)/N)
#             ranks = ranks.leftOuterJoin(ranks_u).map(lambda line: (line[0], line[1][0] if line[1][1] is None else line[1][1]))
            
            if debug_mod == 1:
                ut.printRDD(ranks)
        return ranks.sortByKey()
#         for (link, rank) in ranks.collect():
#             print("%s has rank: %s." % (link, rank))

    def statistics_compute_weighted(self, D, Iter, d, debug_mod):
        max_src, max_dst = D.reduce(lambda x, y: (max(x[0], y[0]), max(x[1], y[1])))
        print(max_src, max_dst)
        
        N = float(max(max_src, max_dst))
        src_nodes = D.map(lambda x: (x[0], 0)).distinct()
        dst_nodes = D.map(lambda x: (x[1], 0)).distinct()
        src_only = src_nodes.subtractByKey(dst_nodes).cache()
        
        ranks = D.map(lambda line: line[0]).union(D.map(lambda line: line[1])).distinct().map(lambda line: (line, 1/N)).cache()
        
        D = D.map(lambda x: (x[0], (x[1], x[2])));
        
        for iter in xrange(Iter):
            if debug_mod == 1:
                print(iter)
                ut.printRDD(D)
            contribs = D.groupByKey().join(ranks).flatMap(lambda url_urls_rank: self.computeContribs_weighted(url_urls_rank[1][0], url_urls_rank[1][1]))
            contribs = contribs.union(src_only)
            
            if debug_mod == 1:
                ut.printRDD(contribs)
                
            ranks = contribs.reduceByKey(add).mapValues(lambda rank: rank * d + (1-d)/N)
#             ranks = ranks.leftOuterJoin(ranks_u).map(lambda line: (line[0], line[1][0] if line[1][1] is None else line[1][1]))
            
            if debug_mod == 1:
                ut.printRDD(ranks)
        return ranks.sortByKey()

    '''
    binCenter returns the center of bin given boundary of bin
    binBoundary: array([double])
    centers: array([double])
    '''
    def binCenter(self,binBoundary):
        length = len(binBoundary)
        centers = [0] * (length-1)
        for idx in range(len(centers)):
            centers[idx] = binBoundary[idx] + (binBoundary[idx+1] - binBoundary[idx])/2.0

        return centers

    def pr_vs_count(self, pr, binNum):
        pr = pr.map(lambda x: x[1])
        histList = pr.histogram(binNum)
        centers = self.binCenter(histList[0])
        return [centers, histList[1]]
        
        
