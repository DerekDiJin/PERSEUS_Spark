'''
Created on Feb 26, 2017

@author: DiJin
'''
import sys
import os
import re
import random
import numpy as np
import argparse

    
class Configurations:
    
    '''
    Add here fore more statistics
    '''
    
    def __init__(self):
#         parser = argparse.ArgumentParser()

        self._debug = 0 # default: 0
        self._weighted = 0 # default: 0
        self._deg_in = 1  # default: 1
        self._deg_out = 1  # default: 1
        self._deg_total = 1  # default: 0
        
        self._pr = 1  # default: 1
        self._svd = 1
        
        # bivariate distribution, total
        self._totaldeg_vs_count = 1 # default: 1
        self._pr_vs_count = 1 # default: 1
        self._totaldeg_vs_pr = 1 # default: 1
        
        # aggregate the result
        self._aggregate_result = 1
        
    def getDebug(self):
        return self._debug
    
    def isWeighted(self):
        return self._weighted
        
    def getIndeg(self):
        return self._deg_in
    
    def getOutdeg(self):
        return self._deg_out
    
    def getTotaldge(self):
        return self._deg_total
    
    def getPR(self):
        return self._pr
        
    def getTotalDeg_vs_Count(self):
        return self._totaldeg_vs_count

    def getPR_vs_Count(self):
        return self._pr_vs_count

    def getTotalDeg_vs_PR(self):
        return self._totaldeg_vs_pr
    
    def getAggregateResult(self):
        return self._aggregate_result
    
    def getSVD(self):
        return self._svd
