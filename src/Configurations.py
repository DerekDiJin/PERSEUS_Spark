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
        self._deg_in = 1
        self._deg_out = 1
        self._deg_total = 1
        
        
        
    def getIndeg(self):
        return self._deg_in
    
    def getOutdeg(self):
        return self._deg_out
    
    def getTotaldge(self):
        return self._deg_total
    
    
        
        