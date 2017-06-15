'''
Project PERSEUS Computing Clustering Coefficient. (CCFinder)
'''

import sys
import os
import re
import random
import math
import numpy as np
import pyspark
from operator import add
from pyspark import SparkConf, SparkContext
import Utility as ut

class ClusteringCoefficient:

    def __init__(self):
        self._description = 'In Clustering Coefficient'

    def appendDeg(self, nodeId, nbors):
        """
        append degree of currentNode to nborID: (nodeId, (nbor_nodeId, degree of nodeId))
        """
        deg = len(nbors)
        for nbor in nbors:
            yield (nbor,(nodeId,deg))

    def fonlFilter(self, nodeId, nbor_deg):
        """construct fonl data structure"""
        nodeIdDeg = len(nbor_deg)
        tempList = []
        for ele in nbor_deg:
            if ((ele[1] > nodeIdDeg) or ((ele[1] == nodeIdDeg) and (ele[0] > nodeId))):
                tempList.append([ele[0],ele[1]])
        tempList.sort(key=lambda x:(x[1],x[0]))
        return (nodeId, nodeIdDeg, [ele[0] for ele in tempList])

    def constructFonl(self, inData):
        """
        Construct FONL data structure
        :inData: input src-dst edge data pair
        :return: RDD object in FONL structure
        """
        inData = inData.union(inData.map(lambda x: (x[1],x[0]))).distinct()

        nodeNbor_deg = inData.groupByKey().flatMap(lambda nodeNbor: self.appendDeg(nodeNbor[0],nodeNbor[1])).groupByKey()

        outputRDD = nodeNbor_deg.map(lambda nNdeg: self.fonlFilter(nNdeg[0], nNdeg[1]))

        return outputRDD

    def generateCandidate(self, nodeId, nbors):
        """
        Generate Candidate List from FONL
        :fonlRDD: RDD obj in FONL structure
        :return candidate list in RDD
        """
        length = len(nbors)
        if length > 1:
            for idx, itm in enumerate(nbors):
                if idx != (length-1):
                    vertex = nbors[idx]
                    sublist = nbors[(idx+1):]
                    yield (vertex, (nodeId, sublist))


    def triangleCounting(self, nodeId, candVal, fonlVal):
        """
        Count the number of triangles in graph
        :candRDD: candidate list generated from fonlRDD
        :fonlRDD: RDD obj in FONL structure
        :return RDD obj. recording # of triangles for each node (nodeID, num)
        """
        for ele in candVal:
           tri = list(set(fonlVal).intersection(set(ele[1])))
           num = len(tri)
           for itm in tri:
               yield (itm, 1)
           yield (ele[0], num)
           yield (nodeId, num)

    def ccfinding(self, inData):
        """
        Computing LCC for each node
        :inData input src-dst edge data pair
        :return RDD obj. recording clustering coefficient for each node (nodeID, cc)
        """
        """ Calculate FONL structure """
        num = inData.flatMap(lambda x: (x[0], x[1])).max()
        print("num of nodes is " + str(num))
        fonlRDD = self.constructFonl(inData).cache()
#        ut.printRDD(fonlRDD)

        """ Calculate candidate list """
        candRDD = fonlRDD.flatMap(lambda x: self.generateCandidate(x[0],x[2])).cache()
        candRDD = candRDD.groupByKey()
#        print("candRDD (aft group)")
#        ut.printRDD(candRDD.map(lambda x: (x[0], list(x[1]))))

        """ Calculate num of triangles for each node """
        tempRDD = fonlRDD.filter(lambda x: len(x[2]) > 0).map(lambda x: (x[0], x[2]))
#        print("joined_candRDD")
#        ut.printRDD(candRDD.join(tempRDD).mapValues(list).map(lambda x: (x[0], list(x[1][0]), x[1][1])))
        triRDD = candRDD.join(tempRDD).flatMap(lambda x: self.triangleCounting(x[0], x[1][0], x[1][1])).reduceByKey(add)
#        print("triRDD")
#        ut.printRDD(triRDD)

        """ Calculate the denuminator of each node """
        denum = fonlRDD.map(lambda x: (x[0], 0.5*x[1]*(x[1]-1)))

        """ Calculate local clustering coefficient for each node """
        lccRDD = triRDD.join(denum).map(lambda x: (x[0], x[1][0]/x[1][1]))
#         print("local clustering coefficient for each node is:")
#         ut.printRDD(lccRDD)
        acc = lccRDD.map(lambda x: x[1]).sum()/num
        curNodes = lccRDD.map(lambda x: (x[0], 0))
        Nodes = inData.map(lambda x: (x[0], 0)).subtractByKey(curNodes)
#         ut.printRDD(Nodes)
        lccRDD = lccRDD.union(Nodes).distinct().sortByKey()
#         ut.printRDD(lccRDD)
        
        return lccRDD, acc

    def extreme_compute(self, lccRDD):
        
        cc_min = lccRDD.map(lambda x: (x[1])).min()
        cc_max = lccRDD.map(lambda x: (x[1])).max()
        
        return cc_min, cc_max

#    def findIndex(self, value, min_value, max_value, N, centers):
#        if value == max_value:
#            return centers[N-1]
#        else:
#            interval = max_value - min_value
#            grid = interval / N
#            index = int(math.floor( (value-min_value) / grid))
#            if index >= N-1:
#                index = N-1
#            return centers[index]

#    def binCenter(self,binBoundary):
#        length = len(binBoundary)
#        centers = [0] * (length-1)
#        for idx in range(len(centers)):
#            centers[idx] = binBoundary[idx] + (binBoundary[idx+1] - binBoundary[idx])/2.0
#        return centers

    def cc_vs_count(self, cc, binNum):
        cc = cc.map(lambda x: x[1])
        histList = cc.histogram(binNum)
        centers = ut.binCenter(histList[0])
        return [centers, histList[1]]

    

