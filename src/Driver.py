'''
Created on Feb 26, 2017

@author: DiJin
@author: Haoming Shen (Weighted in/out degree)
'''
import sys
import os
import re
import random
import numpy as np
import pyspark
from pyspark import SparkConf, SparkContext

import Utility as ut

from Configurations import Configurations

from Degrees import Degrees
from PageRank import PageRank
    
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
    
def parse_raw_lines_weighted(line):
    if line.startswith('#'):
        return -1, -1
    else:
        line = line.strip('\r\n')
        parts = re.split('\t| ', line)
        x_id = int(parts[0])
        y_id = int(parts[1])
        weight = float(parts[2])
        return x_id, y_id, weight
    

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



#########################################################################################

if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print ('Usage: spark-submit Degrees.py <input_filepath> <output_filepath>')
        sys.exit(-1)

    data_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    
    # default settings
    initialID = 1
    
    mod = 'local'
    
    # setting up spark
    
    if mod == 'local':
        os.environ["SPARK_HOME"] = "/Users/DiJin/BigData/spark-1.6.0-bin-hadoop2.6"
        # configure the Spark environment
        sparkConf = pyspark.SparkConf().setAppName("PERSEUS_Spark")\
        .setMaster("local")

    elif mod == 'yarn':
        os.environ['PYSPARK_PYTHON'] = '/sw/lsa/centos7/python-anaconda2/201607/bin/python'
        sparkConf = pyspark.SparkConf().setMaster("yarn-client").setAppName("LP")
        sparkConf.set("spark.executor.heartbeatInterval","3600s")
    #     .setMaster("local")                                                            # <<<---
        
    else:
        sys.exit('mode error: only local and yarn are accepted.')
    
    
    sc = pyspark.SparkContext(conf = sparkConf)
    '''
    assume the nodes in the input are re-ordered and no duplication
    load data file with format:
    <src>    <dst>
    remove all self loops and titles
    '''
    
    graph_statistics = Configurations()
    debug_mod = graph_statistics.getDebug()
    
    if graph_statistics.isWeighted() == 0:
        lines = sc.textFile(data_file_path)
        tempD = lines.map(lambda line: parse_raw_lines(line)).cache()
        D = tempD.filter(lambda x: x[0] > 0 and x[0] != x[1]).cache()
        
    elif graph_statistics.isWeighted() == 1:
        lines = sc.textFile(data_file_path)
        tempD_w = lines.map(lambda line: parse_raw_lines_weighted(line)).cache()
        D_w = tempD_w.filter(lambda x: x[0] > 0 and x[0] != x[1]).cache()
    
#     ut.printRDD(D_w.groupByKey())
    

    if graph_statistics.isWeighted() == 0:
    
        '''
        Degrees
        '''
        deg = Degrees()
        
        if graph_statistics.getOutdeg():
            output_rdd = deg.statistics_compute(D, 'out')
            
            # generate outputs to hdfs
            temp = output_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'out_degree')
            
        if graph_statistics.getIndeg():
            output_rdd = deg.statistics_compute(D, 'in')
            
            # generate outputs to hdfs
            temp = output_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'in_degree')
            
        if graph_statistics.getTotaldge():
            output_rdd = deg.statistics_compute(D, 'total')
            
            # generate outputs to hdfs
            temp = output_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'total_degree')
            
        if graph_statistics.getTotalDeg_vs_Count():
            output_rdd = deg.statistics_compute(D, 'total')
            deg_vs_count_rdd = deg.deg_vs_count(output_rdd)
            
            # generate outputs to hdfs
            temp = deg_vs_count_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'deg_vs_count')
            
        '''
        PageRank
        '''       
        pr = PageRank() 
        
        if graph_statistics.getPR():
            output_rdd = pr.statistics_compute(D, 19, 0.85, debug_mod)
            
            # generate outputs to hdfs
            temp = output_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'pagerank')
            
        if graph_statistics.getPR_vs_Count():
            pr_rdd = pr.statistics_compute(D, 19, 0.85, debug_mod)
            [centers, counts] = pr.pr_vs_count(output_rdd, 1000)
            centers = sc.parallelize(centers)
            counts = sc.parallelize(counts)
            pr_vs_count = centers.zip(counts)
      
            # generate outputs to hdfs
            temp = pr_vs_count.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'pr_vs_count')
            
        if graph_statistics.getTotalDeg_vs_PR():
            total_degree_rdd = deg.statistics_compute(D, 'total')
            pr_rdd = pr.statistics_compute(D, 19, 0.85, debug_mod)
            total_degree_vs_pr_rdd = total_degree_rdd.join(pr_rdd).map(lambda x: x[1])
            
            temp = total_degree_vs_pr_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'total_degree_vs_pr')
            
            

    elif graph_statistics.isWeighted() == 1:
        '''
        Degrees
        '''
        deg = Degrees()
        if graph_statistics.getOutdeg():
            print("Starts computing weighted_out degree...")
            output_rdd = deg.statistics_compute(D_w, 'weighted_out')
            print(output_rdd)
            
            # generate outputs to hdfs
            temp = output_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'out_degree_weighted')
            
        if graph_statistics.getIndeg():
            output_rdd = deg.statistics_compute(D_w, 'weighted_in')
            
            # generate outputs to hdfs
            temp = output_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'in_degree_weighted')
           
        if graph_statistics.getTotaldge():
            output_rdd = deg.statistics_compute(D_w, 'weighted_total')
            
            # generate outputs to hdfs
            temp = output_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'total_degree_weighted')
            
        if graph_statistics.getTotalDeg_vs_Count():
            [centers, counts] = deg.deg_vs_count_weight(output_rdd, 1000)
            centers = sc.parallelize(centers)
            counts = sc.parallelize(counts)
            total_degree_vs_count_weighted_rdd = centers.zip(counts)
            
            # generate outputs to hdfs
            temp = total_degree_vs_count_weighted_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'total_degree_vs_count_weighted')
   
        '''
        PageRank
        '''       
        pr = PageRank() 
        
        if graph_statistics.getPR():
            output_rdd = pr.statistics_compute_weighted(D_w, 32, 0.85, debug_mod)
            
            # generate outputs to hdfs
            temp = output_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'pagerank_weighted')
            
        
        if graph_statistics.getPR_vs_Count():
            pr_rdd = pr.statistics_compute(D, 32, 0.85, debug_mod)
            [centers, counts] = pr.pr_vs_count(output_rdd, 1000)
            centers = sc.parallelize(centers)
            counts = sc.parallelize(counts)
            pr_vs_count = centers.zip(counts)
      
            # generate outputs to hdfs
            temp = pr_vs_count.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'pr_vs_count_weighted')
        
        
        if graph_statistics.getTotalDeg_vs_PR():
            total_degree_rdd = deg.statistics_compute(D, 'total')
            pr_rdd = pr.statistics_compute(D, 32, 0.85, debug_mod)
            total_degree_vs_pr_rdd = total_degree_rdd.join(pr_rdd).map(lambda x: x[1])
            
            temp = total_degree_vs_pr_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'total_degree_vs_pr_weighted')
            
    
#      
#     Edges_rdd = sc.parallelize(Edges)
#     temp = Edges_rdd.map(lambda x: ",".join(map(str,x))).coalesce(1)
#     temp.saveAsTextFile(output_file_path+'edges')
