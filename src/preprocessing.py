"""
The dat sets preprocessing by joining corresponding matrices

Created by Iaroslav Omelianenko
"""
import argparse
import sys
import time 

import numba

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix

@numba.jit
def numbaStrCompare(left, right):
    result = True
    for i in range(32):
        if (left[i] != right[i]):
            result = False
            break
            
    # Strings equal        
    return result

@numba.jit
def findIndex(query, target):
    n_target = len(target)
    result = -1
    for j in range(n_target):
        if numbaStrCompare(query, target[j]):
            result = j
            break

    return result

@numba.jit
def findIndexes(left, target):
   n_res = len(left)
   result = np.empty(n_res, dtype='int32')
   for i in range(n_res):
       index = findIndex(left[i], target)
       result[i] = index

   return result
   
def buildFBLikesDataSet(users_csv, likes_csv, users_likes_csv):
    """
    Method to build data set based on FB user's likes
    Parameters:
        users_csv : the path to the CSV file with users data
        likes_csv : the path to the CSV file with FB likes data
        users_likes_csv : the path to the file with users<->likes associations
    Returns:
        matrix : the sparse matrix with [users, likes]
        users_df : the users DataFrame
        likes_df : the likes DataFrame
    """
    users_df = pd.read_csv(users_csv)
    likes_df = pd.read_csv(likes_csv)
    users_likes_df = pd.read_csv(users_likes_csv)
    
    print '\n------------------------\nUsers:\n%s' % users_df.describe()
    print '\n------------------------\nLikes:\n%s' % likes_df.describe()
    

    start_time = time.time()
    ul_size = 100000#len(users_likes_df)
    print '\nStart building users/likes sparse matrix with size: %d' % ul_size
    matrix = dok_matrix((len(users_df), len(likes_df)), dtype=np.int16)
    
    for i in range(ul_size):
        userid = users_likes_df['userid'][i]
        likeid = users_likes_df['likeid'][i]
        user_idx = users_df[users_df['userid'] == userid].index[0]
        like_idx = likes_df[likes_df['likeid'] == likeid].index[0]
        matrix[user_idx, like_idx] = 1 # FB user can issue like only once
    
    """
    users_likes_df_part = users_likes_df.head(ul_size)
    users_idx = findIndexes(users_likes_df_part['userid'].values, users_df['userid'].values)
   
    print '\n------------------------\nIndices:\n%s' % (users_likes_df_part, users_idx)
    """
                 
    build_time = time.time() - start_time
    print '\n\nSparse matrix build complete in: %0.2f sec with final size: %d' % (build_time, matrix.getnnz())
            
           
    # trimming data
    m_shape = matrix.shape
    print '\nResulting matrix shape: (%d, %d)' % (m_shape[0], m_shape[1])       
           
    return matrix, users_df, likes_df
   
def processFBLikesDS(args):
    if args.fbin == None:
        print "No input files provided for FB likes dataset"
        sys.exit(1)
    elif len(args.fbin) != 3:
        print "Wrong number of input files provided: %d" % len(args.fbin)
        sys.exit(1)
    
    users_csv = args.fbin[0]
    likes_csv = args.fbin[1]
    users_likes_csv = args.fbin[2]
            
    print 'Pre-processing FB likes data set\nusers file: %s\nlikes file: %s\nusers-likes file:%s\n' \
            % (users_csv, likes_csv, users_likes_csv)

    matrix, users_df, likes_df = buildFBLikesDataSet(users_csv, likes_csv, users_likes_csv)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the name of data set to preprocess")
    parser.add_argument("--fbin", nargs="+", help="the list of FB data set input files (users.csv, likes.csv, users_likes.csv)")
    args = parser.parse_args()

    print args    
    
    # Read arguments
    dataset = args.dataset
    if dataset == 'fblikes':
        processFBLikesDS(args)
        
   
    
    
