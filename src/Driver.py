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
from pyspark.sql import SQLContext
from pyspark.sql import Row

from pyspark.mllib.linalg import Vectors
# from linalg import Vectors
# from pyspark.mllib.linalg.distributed import RowMatrix
from linalg.distributed import RowMatrix


import Utility as ut

from Configurations import Configurations

from Degrees import Degrees
from PageRank import PageRank
from SVD import SVD
from ClusteringCoefficient import ClusteringCoefficient

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
    v_dim = 10
    Iter = 10
    
    mod = 'local'
    
    # setting up spark
    
    if mod == 'local':
        os.environ["SPARK_HOME"] = "/Users/DiJin/BigData/spark-2.1.1-bin-hadoop2.6"
        # configure the Spark environment
        sparkConf = pyspark.SparkConf().setAppName("PERSEUS_Spark")\
        .setMaster("local").set("spark.executor.memory","4g").set("spark.driver.memory","2g")

    elif mod == 'yarn':
        #os.environ['PYSPARK_PYTHON'] = '/sw/lsa/centos7/python-anaconda2/201607/bin/python'                                                                  
        sparkConf = pyspark.SparkConf().setAppName("LP").set("spark.executor.memory","2g").set("spark.driver.memory","2g").set("spark.yarn.executor.memoryOve\
rhead", "1g")
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
            
            
        if graph_statistics.getSVD() == 1:   
        
            svd = SVD()
            x_max = D.map(lambda x: x[0]).max() - 1
            y_max = D.map(lambda x: x[1]).max() - 1
            print (x_max, y_max)
            
            if x_max > y_max:
                D = D.map(lambda x: (x[1], x[0])).cache()
                temp = x_max
                x_max = y_max
                y_max = temp
            
            adj_list = ut.edgelist2Adj(D, x_max, y_max)
            adj_list_rdd = sc.parallelize(adj_list).cache()
            
                
            mat = RowMatrix(adj_list_rdd)
            
            v_dim = min(y_max+1, 10)
            # Compute the top 5 singular values and corresponding singular vectors.
            svd_result = svd.statistics_compute(mat, v_dim, True)
            U = svd_result.U       # The U factor is a RowMatrix.
            s = svd_result.s       # The singular values are stored in a local dense vector.
            V = svd_result.V       # The V factor is a local dense matrix.
            
    #         ut.printRDD(U.rows)
    
#  -->           fOut = open(output_file_path+'S', 'w')
#             for item in s:
#                 fOut.write(str(item) + '\n')
#             fOut.close()
            
#             fOut = open(output_file_path+'V', 'w')
            v_array = V.toArray()
            
            temp = []
            id = 1
            for ele in v_array:
                array_cur = np.insert(ele, 0, id)
                temp.append(array_cur.astype(float))
                id = id + 1
            v_array = temp
            
            v_tuple = tuple( map(tuple, v_array) )
            # write to V file
#  -->           for ele in v_array:
#                 newStr = str(ele[0])
#                 for e in ele[1:]:
#                     newStr = newStr + '\t' + str(e)
#                 fOut.write(newStr + '\n')
#             fOut.close()
            
            v_tuple_rdd = sc.parallelize(v_tuple)
#             ut.printRDD(v_tuple_rdd)
#             id_rdd = sc.parallelize([i for i in range(1, max(x_max, y_max)+2)])
            v_rdd = v_tuple_rdd.map(lambda l: Row(*[str(x) for x in l]))
                    
        
            
        if graph_statistics.combineResult() == 1:
            
            sqlContext = SQLContext(sc)
            
            temp = D.map(lambda x: (x[0], x[1], 1)).map(ut.toTSVLine).coalesce(1)
            temp.saveAsTextFile(output_file_path+'edges')
            
            '''
            combine true statistics
            '''
            """ degree """
            total_degree_rdd = deg.statistics_compute(D, 'total')
            total_degree_df = sqlContext.createDataFrame(total_degree_rdd, ['nodeid', 'degree'])
            
            deg_vs_count_rdd = deg.deg_vs_count(total_degree_rdd)
            deg_vs_count_df = sqlContext.createDataFrame(deg_vs_count_rdd, ['degree', 'count'])
            
            part1_df = total_degree_df.join(deg_vs_count_df, total_degree_df.degree == deg_vs_count_df.degree).drop(deg_vs_count_df.degree)

            """ pagerank """
            pr_rdd = pr.statistics_compute(D, Iter, 0.85, debug_mod)
            pr_df = sqlContext.createDataFrame(pr_rdd, ['nodeid', 'pagerank'])         
            
            [centers, counts] = pr.pr_vs_count(pr_rdd, N)
            centers_rdd = sc.parallelize(centers)
            counts_rdd = sc.parallelize(counts)
            pr_vs_count_rdd = centers_rdd.zip(counts_rdd)
            pr_vs_count_df = sqlContext.createDataFrame(pr_vs_count_rdd, ['pagerank_t', 'pagerank_t_count'])
            
            pr_min, pr_max = pr.extreme_compute(pr_rdd)
            pr_pr_t_rdd = pr_rdd.map(lambda x: (x[1], pr.findIndex(x[1], pr_min, pr_max, N, centers))).distinct()
            pr_pr_t_df = sqlContext.createDataFrame(pr_pr_t_rdd, ['pagerank', 'pagerank_t'])
#             
            part2_df_temp = pr_df.join(pr_pr_t_df, pr_df.pagerank == pr_pr_t_df.pagerank).drop(pr_pr_t_df.pagerank)
            part2_df = part2_df_temp.join(pr_vs_count_df, part2_df_temp.pagerank_t == pr_vs_count_df.pagerank_t).drop(pr_vs_count_df.pagerank_t)

            deg_pr_df = part1_df.join(part2_df, part1_df.nodeid == part2_df.nodeid).drop(part2_df.nodeid)
#             deg_pr_df.show()
#             deg_pr_df.sort("nodeid", ascending=True).coalesce(1).write.csv(output_file_path+'combined', 'overwrite')
#             

            """ cluster coeff """
            cc = ClusteringCoefficient()
            cc_rdd, acc = cc.ccfinding(D)
            cc_min, cc_max = cc.extreme_compute(cc_rdd)
            [centers, counts] = cc.cc_vs_count(cc_rdd, N)
            centers_rdd = sc.parallelize(centers)
            counts_rdd = sc.parallelize(counts)
            cc_vs_count = centers_rdd.zip(counts_rdd)
            cc_df = sqlContext.createDataFrame(cc_rdd, ['nodeid', 'clusCoeff'])
            # cc_df.show()
            cc_vs_cnt_df = sqlContext.createDataFrame(cc_vs_count, ['clusCoeff_t', 'cc_t_count'])
            # cc_vs_cnt_df.show()
            cc_cc_t_rdd = cc_rdd.map(lambda x: (x[1], ut.findIndex(x[1],cc_min,cc_max,N,centers))).distinct()
            cc_cc_t_df = sqlContext.createDataFrame(cc_cc_t_rdd, ['clusCoeff', 'clusCoeff_t'])

            part3_df_temp = cc_df.join(cc_cc_t_df, cc_df.clusCoeff == cc_cc_t_df.clusCoeff).drop(cc_cc_t_df.clusCoeff)
            part3_df = part3_df_temp.join(cc_vs_cnt_df, part3_df_temp.clusCoeff_t == cc_vs_cnt_df.clusCoeff_t).drop(cc_vs_cnt_df.clusCoeff_t)
            
            """ degree + pagerank + cluster coeff """
            deg_cc_df = deg_pr_df.join(part3_df, deg_pr_df.nodeid == part3_df.nodeid).drop(part3_df.nodeid)
#             deg_cc_df.show()
            
            """ SVD """
            v_dim_list = ['nodeid']
            for i in xrange(v_dim):
                v_dim_str = 'v_' + str(i+1)
                v_dim_list.append(v_dim_str)
  
            v_df = sqlContext.createDataFrame(v_rdd, v_dim_list)
            v_df = v_df.select(v_df.nodeid.cast('int'), v_df.v_1, v_df.v_2, v_df.v_3, v_df.v_4, v_df.v_5, v_df.v_6, v_df.v_7, v_df.v_8, v_df.v_9, v_df.v_10)
             
            v_df = svd.addApprox(sc, v_df, v_rdd, N)
#             v_df.show()
#                 ut.printRDD(val_val_t_rdd)
  
            all_statistics_df = deg_cc_df.join(v_df, deg_cc_df.nodeid == v_df.nodeid).drop(v_df.nodeid)
            all_statistics_df.sort("nodeid", ascending=True).coalesce(1).write.csv(output_file_path+'combined', 'overwrite')
#             all_statistics_df.show()
#             
            '''
            aggregate true statistics for separate plots
            '''
            # plot 1: deg vs count
            plot1_df = all_statistics_df.select(['degree', 'count']).distinct()
            plot1_df.coalesce(1).write.csv(output_file_path+'/plots/deg_vs_count', 'overwrite')
             
             
            plot2_df = all_statistics_df.groupby(['degree', 'pagerank_t']).count()
            plot2_df.coalesce(1).write.csv(output_file_path+'/plots/deg_vs_pr', 'overwrite')
             
            plot3_df = all_statistics_df.select(['pagerank_t', 'pagerank_t_count']).distinct()
            plot3_df.coalesce(1).write.csv(output_file_path+'/plots/pr_vs_count', 'overwrite')
             
            plot4_df = all_statistics_df.select(['degree', 'count', 'pagerank_t', 'pagerank_t_count', 'clusCoeff_t', 'cc_t_count', 'v_1_t', 'v_2_t', 'v_3_t', 'v_4_t', 'v_5_t', 'v_6_t', 'v_7_t', 'v_8_t', 'v_9_t', 'v_10_t']).distinct()
            plot4_df.coalesce(1).write.csv(output_file_path+'/plots/all', 'overwrite')
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
            

    sc.stop()
        
#         ut.printRDD(v_array_rdd)
        
#         if graph_statistics.combineResult() == 1:
#             all_rdd = combined_rdd.map(lambda x: (x[0], (x[1], x[2], x[3], x[4]))).join(v_array_rdd).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], '\t'.join(str(d) for d in x[1][1])))# <<<---
# #             ut.printRDD(all_rdd)
#             temp = all_rdd.map(ut.toTSVLine).coalesce(1)
#             temp.saveAsTextFile(output_file_path+'all')
          
#         collected = U.rows.collect()
#         print("U factor is:")
#         for vector in collected:
#             print(vector)
#         print("Singular values are: %s" % s)
#         print("V factor is:\n%s" % V)
        
        
        
#
#     Edges_rdd = sc.parallelize(Edges)
#     temp = Edges_rdd.map(lambda x: ",".join(map(str,x))).coalesce(1)
#     temp.saveAsTextFile(output_file_path+'edges')
