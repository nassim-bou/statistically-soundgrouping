import sys
import pandas as pd
import numpy as np
from itertools import product 
import itertools
import datetime
from multiprocessing import Pool

from source_code.core.hypothesis_evaluation.test_handler import baseline_test_groups
import source_code.core.hypothesis_evaluation.generate_results as gen_results
from source_code.core.hypothesis_evaluation.generate_graphs import *

from source_code.models.models import get_data
from source_code.utils.tools import add_index
import altair as alt

import pickle

alt.data_transformers.disable_max_rows()

dataset = 'MovieLens'
alphas = 0.05
support = 20
dimension = 'rating'
grouping_type = [['year'], ['year'], ['year']]
grouping_value = [["80's"], ["80's"], ["80's"]]
agg_type = 'anova'
test_args = ['Anova']
periods_start = [datetime.datetime(2001, 7, 1, 0, 0), datetime.datetime(2001, 7, 8, 0, 0), datetime.datetime(2001, 7, 15, 0, 0)]
period_arg = [0, 5]
top_ns=[5, 10, 15, 20, 50, 100, 500, -1]
period_type='time'
num_hyps = [0.1, 0.3, 0.5, 0.7, 0.9, 1]

r = 'R4'

methods = [0,1,4,5,6,7,8,9]

parameters = f"dataset = '{dataset}'\nalphas = {alphas}\nsupport = {support}\ndimension = '{dimension}'\n"
parameters = parameters + f"grouping_type = {grouping_type}\ngrouping_value = {grouping_value}\n"
parameters = parameters + f"agg_type = '{agg_type}'\ntest_args = {test_args}\nperiods_start = {periods_start}\n"
parameters = parameters + f"period_arg = {period_arg}\ntop_ns={top_ns}\nperiod_type='{period_type}'\n"
parameters = parameters + f"num_hyps = {num_hyps}"

fil = open(f'../experiments/{dataset}/{r}/parameters','w')
fil.write(parameters)
fil.close()

d = {0:'TRAD',1:'COVER_G',2:'coverage_Side_1',3:'coverage_Side_2', 4:'COVER_⍺',5:'β-Farsighted',6:'γ-Fixed',7:'ẟ-Hopeful',\
8:'Ɛ-Hybrid',9:'Ψ-Support'}
 
def worker(num):
    for k in range(num): 
        stats, res, p_values = baseline_test_groups(top_ns, num_hyps, dataset, methods, period_type, periods_start, period_arg, grouping_type,\
        grouping_value, agg_type, dimension, test_args, support, alphas, r, verbose=True)

        if stats is None:
            print('No Results')
        else:
            samples = stats.keys()

        a_file = open(f"../experiments/{dataset}/{r}/stats_{r}_{k}.pkl", "wb")
        pickle.dump(stats, a_file)
        a_file.close()

        a_file = open(f"../experiments/{dataset}/{r}/res_{r}_{k}.pkl", "wb")
        pickle.dump(res, a_file)
        a_file.close()

        params = {
            'TRAD':["fdr_by", "fdr_b"],
            'COVER_G':["fdr_by", "fdr_b"],
            'COVER_⍺':[20,50,100,200,300,500],
            'β-Farsighted':[0.25, 0.5, 0.75, 0.9],
            'γ-Fixed':[20,50,100,200,300,500],
            'ẟ-Hopeful':[20,50,100,200,300,500],
            'Ɛ-Hybrid':[ (0.25,500,500), (0.5,500,500), (0.75,500,500), (0.25,100,1000), (0.5,100,100), (0.75,100,100),\
            (0.25,200,200), (0.5,200,200), (0.75,200,200) ],
            'Ψ-Support':[1/2, 1/3, 1/4, 1/5, 1/6]
        }

        #p_values.to_csv(f'../experiments/{dataset}/{r}/final_p_values_{r}.csv',index=False)

        samples = res.keys()

        df = pd.DataFrame(columns=['group1','group2','coverage_gained_1','coverage_gained_2','p-value','wealth','alpha_j', 'nb_refus_before','algorithm','param','top','sample'])
        
        for sample in samples:
            top_n = res[sample].keys()
            
            for tp in top_n:
                res_local = res[sample][tp]
                mthd = res_local.keys()
                
                for mtd in mthd:
                    for i in range(len(res_local[mtd])):
                        df2 = res_local[mtd][i].copy()
                        
                        if mtd == 'traditional':
                            df2['wealth']=0
                            df2['alpha_j']=0
                            df2['nb_refus_before']=0
                        
                        df2['algorithm'] = mtd
                        df2['param'] = str(params[mtd][i])
                        df2['top'] = tp
                        df2['sample'] = sample
                        df = pd.concat([df,df2])

        df.to_csv(f'../experiments/{dataset}/{r}/result_group_descriptions_{r}_{k}.csv',index=False)

        samples = stats.keys()

        df = pd.DataFrame(columns=['#AllCandidates','#IndepCandidates','#ResultsNoAdjustment','GroupingTime','#Results','Min_p_values','Max_p_values',\
        'Sum_p_values','Cov_total','p_value_adjustement_time','algorithm','param','top','sample'])

        for sample in samples:
            top_n = stats[sample].keys()
            
            for tp in top_n:
                res_local = stats[sample][tp]
                
                for i,mtd in enumerate(mthd):
                    df2 = res_local[i].copy()
                    df2 = df2.values[0]
                    
                    df_intermed = np.concatenate((df2[:3],df2[-1:]), axis=0 )
                    df2 = df2[6:-1]
                    
                    col_in_mtd = 8
                    
                    for j in range( int(len(df2)/col_in_mtd) ):
                        df_intermed_2 = df2[j*col_in_mtd:(j+1)*col_in_mtd]
                        
                        par = str(params[mtd][j])
                        
                        df.loc[len(df)] = list(df_intermed)+list(df_intermed_2[:4])+list(df_intermed_2[6:])+[mtd,par,tp,sample]

        df.to_csv(f'../experiments/{dataset}/{r}/result_statistics_{r}_{k}.csv',index=False)

worker(num=10)