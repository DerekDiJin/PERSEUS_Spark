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
from pyspark.sql.functions import countDistinct

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
    N = 1000
    Iter = 10
    
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
            out_deg_rdd = deg.statistics_compute(D, 'out')
            
            # generate outputs to hdfs
            temp = out_deg_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'out_degree')
            
        if graph_statistics.getIndeg():
            in_deg_rdd = deg.statistics_compute(D, 'in')
            
            # generate outputs to hdfs
            temp = in_deg_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'in_degree')
            
        if graph_statistics.getTotaldge():
            total_deg_rdd = deg.statistics_compute(D, 'total')
            
            # generate outputs to hdfs
            temp = total_deg_rdd.map(ut.toTSVLine).coalesce(1)
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
            pr_rdd = pr.statistics_compute(D, Iter, 0.85, debug_mod)
            
            # generate outputs to hdfs
            temp = pr_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'pagerank')
            
        if graph_statistics.getPR_vs_Count():
            pr_rdd = pr.statistics_compute(D, Iter, 0.85, debug_mod)
            [centers, counts] = pr.pr_vs_count(pr_rdd, N)
            centers = sc.parallelize(centers)
            counts = sc.parallelize(counts)
            pr_vs_count = centers.zip(counts)
      
            # generate outputs to hdfs
            temp = pr_vs_count.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'pr_vs_count')
            
        if graph_statistics.getTotalDeg_vs_PR():
            total_degree_rdd = deg.statistics_compute(D, 'total')
            pr_rdd = pr.statistics_compute(D, Iter, 0.85, debug_mod)
            total_degree_vs_pr_rdd = total_degree_rdd.join(pr_rdd).map(lambda x: x[1])
            
            temp = total_degree_vs_pr_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'total_degree_vs_pr')
            
        if graph_statistics.getAggregateResult() == 1:
            
            total_degree_rdd = deg.statistics_compute(D, 'total')
            pr_rdd = pr.statistics_compute(D, Iter, 0.85, debug_mod)
            
            deg_min, deg_max = deg.extreme_compute(total_degree_rdd)
            pr_min, pr_max = pr.extreme_compute(pr_rdd)
            
            deg_vs_count_rdd = deg.deg_vs_count(output_rdd)
            [centers, counts] = pr.pr_vs_count(pr_rdd, N)
            centers_rdd = sc.parallelize(centers)
            counts_rdd = sc.parallelize(counts)
            pr_vs_count = centers_rdd.zip(counts_rdd)
            
#            nodeid | degree -> nodeid | degree | pr -> degree | pr -> -> degree | pr | count -> \\  
#            pr | degree | count -> pr | degree | count | pr_t -> \\
#            pr_t | pr | degree | count -> pr_t | [ pr | degree | count || pr_t_count] -> degree | count | pr_t | pr_t_count \\
#            
            combined_rdd = total_degree_rdd.join(pr_rdd).map(lambda x: (x[1][0], x[1][1])).join(deg_vs_count_rdd)       \
                .map(lambda x: (x[1][0], (x[0], x[1][1]))).map( lambda x: ( x[0], (x[1][0], x[1][1], pr.findIndex(x[0], pr_min, pr_max, N, centers)) ) )       \
                .map(lambda x: (x[1][2], (x[0], x[1][0], x[1][1]))).join(pr_vs_count).map( lambda x: (x[1][0][1], x[1][0][2], x[0], x[1][1]) )

#            degree | count | pr_t | pr_t_count -> degree | pr_t | dp_count -> \\
#            degree | count | pr_t | pr_t_count -> degree | pr_t || count | pr_t_count | dp_count -> degree | pr_t | dp_count | degree | count | 1 | pr_t | pr_t_count | 1 
            deg_pr_c_rdd = combined_rdd.groupBy(lambda x: (x[0], x[2])).map(lambda x: (x[0], len(x[1])))
            final_rdd = combined_rdd.map( lambda x: ((x[0], x[2]), (x[1], x[3])) ).join(deg_pr_c_rdd).map( lambda x: (x[0][0], x[0][1], x[1][1], x[0][0], x[1][0][0], 1, x[0][1], x[1][0][1], 1) ).distinct()
            ut.printRDD(final_rdd)
#             print(deg_count_rdd)
#             pr_count_rdd = combined_rdd.map(combined_rdd[2], combined_rdd[3])
#             deg_pr_rdd = combined_rdd.map(combined_rdd[0], combined_rdd[2])
            
            temp = final_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'combined')
#             fOut_path = output_file_path+'combined'
#             fOut = open(fOut_path, 'w')
#             for key, value in combined_rdd.items():
#                 fOut.write(str(key) + '\t' + str(value) + '\n')
#             fOut.close()
#                 
            
            
            

    elif graph_statistics.isWeighted() == 1:
        '''
        Degrees
        '''
        deg = Degrees()
        if graph_statistics.getOutdeg():
            print("Starts computing weighted_out degree...")
            out_deg_rdd = deg.statistics_compute(D_w, 'weighted_out')
#             print(output_rdd)
            
            # generate outputs to hdfs
            temp = out_deg_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'out_degree_weighted')
            
        if graph_statistics.getIndeg():
            in_deg_rdd = deg.statistics_compute(D_w, 'weighted_in')
            
            # generate outputs to hdfs
            temp = in_deg_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'in_degree_weighted')
           
        if graph_statistics.getTotaldge():
            total_deg_rdd = deg.statistics_compute(D_w, 'weighted_total')
            
            # generate outputs to hdfs
            temp = total_deg_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'total_degree_weighted')
            
        if graph_statistics.getTotalDeg_vs_Count():
            output_rdd = deg.statistics_compute(D, 'total')
            [centers, counts] = deg.deg_vs_count_weight(output_rdd, N)
            centers = sc.parallelize(centers)
            counts = sc.parallelize(counts)
            deg_vs_count_rdd = centers.zip(counts)
            
            # generate outputs to hdfs
            temp = deg_vs_count_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'total_degree_vs_count_weighted')
   
        '''
        PageRank
        '''       
        pr = PageRank() 
        
        if graph_statistics.getPR():
            pr_rdd = pr.statistics_compute_weighted(D_w, Iter, 0.85, debug_mod)
            
            # generate outputs to hdfs
            temp = pr_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'pagerank_weighted')
            
        
        if graph_statistics.getPR_vs_Count():
            pr_rdd = pr.statistics_compute(D, Iter, 0.85, debug_mod)
            [centers, counts] = pr.pr_vs_count(pr_rdd, N)
            centers = sc.parallelize(centers)
            counts = sc.parallelize(counts)
            pr_vs_count = centers.zip(counts)
      
            # generate outputs to hdfs
            temp = pr_vs_count.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'pr_vs_count_weighted')
        
        
        if graph_statistics.getTotalDeg_vs_PR():
            total_degree_rdd = deg.statistics_compute(D, 'total')
            pr_rdd = pr.statistics_compute(D, Iter, 0.85, debug_mod)
            total_degree_vs_pr_rdd = total_degree_rdd.join(pr_rdd).map(lambda x: x[1])
            
            temp = total_degree_vs_pr_rdd.map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'total_degree_vs_pr_weighted')
            

    
        
        
        
        
#
#     Edges_rdd = sc.parallelize(Edges)
#     temp = Edges_rdd.map(lambda x: ",".join(map(str,x))).coalesce(1)
#     temp.saveAsTextFile(output_file_path+'edges')
