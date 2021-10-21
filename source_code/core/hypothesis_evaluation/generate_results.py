import pandas as pd

def get_fdr_power_results(res_global_stats, res_global, num_hypo, threshold=-1):
    
    fdr_power_all_results = pd.DataFrame(columns=['Metric','Sample','Value','Method','Abv_method'])
    
    fdr_result = dict()
    power_result = dict()
    nb_results = dict()
    
    methods = list(res_global_stats.algorithm.unique())
    
    #For each sample size
    for num in num_hypo:
        
        #Get ground truth
        ground_truth = res_global[ (res_global.algorithm == 'TRAD_BN') & (res_global['sample'] == num * 100) & (res_global.top == -1)]
        
        ground_truth = set(ground_truth.pairs.unique())
                                  
        fdr = []
        power = []
        nb_resu = []

        # Get power and FDR for each method
        for mth in methods:
            loc = res_global[ (res_global.algorithm==mth) & (res_global['sample']==num*100) & (res_global.top==threshold) ]
            
            loc_2 = res_global_stats[ (res_global_stats.algorithm==mth) & (res_global_stats['sample']==num * 100) & \
            (res_global_stats.top==threshold)]
            
            if len(loc.pairs.unique()) != 0:
                power.append( len( ground_truth.intersection(set(loc.pairs.unique())) ) /len(ground_truth) )
            else:
                power.append(0)
                
            if len(loc.pairs.unique()) != 0:
                fdr.append( len(set(loc.pairs.unique()) - ground_truth) /len(loc.pairs.unique()) )
            else:
                fdr.append( 0 )
            
            nb_resu.append( loc_2['#Results'].values[0] )

        power_result[num] = power
        fdr_result[num] = fdr
        nb_results[num] = nb_resu
    
    for key,values in fdr_result.items():
        for j,v in enumerate(values):
            fdr_power_all_results.loc[len(fdr_power_all_results)]=['fdr',f'{int(key*100)}%',v,methods[j],' '*(len(methods)-j)]

    for key,values in power_result.items():
        for j,v in enumerate(values):
            fdr_power_all_results.loc[len(fdr_power_all_results)]=['power',f'{int(key*100)}%',v,methods[j],' '*(len(methods)-j)]

    for key,values in nb_results.items():
        for j,v in enumerate(values):
            fdr_power_all_results.loc[len(fdr_power_all_results)]=['nb_resu',f'{int(key*100)}%',v,methods[j],' '*(len(methods)-j)]
    
    fdr_power_all_results['legend_method'] = fdr_power_all_results['Abv_method']+fdr_power_all_results['Method']
    
    return fdr_power_all_results

def get_coverage_p_values_results(res_global_stats, res_global, num_hypo, threshold=-1):
    cov_all_results = pd.DataFrame(columns=['Coverage','p_values','Method','Abv_method'])
    
    cumul_p_val = dict()
    coverage_results = dict()
    
    if isinstance(num_hypo,int) or isinstance(num_hypo,float):
        num = num_hypo
    else:
        num = list(num_hypo)[-1]
    
    tot_cov = []
    sum_p = []
    
    methods = list(res_global_stats.algorithm.unique())

    #Get the coverage and p-value of each method
    for mth in methods:
        loc = res_global[ (res_global.algorithm==mth) &(res_global['sample']==num*100) & (res_global.top==threshold) ]
        
        loc = loc[['p-value','pairs','coverage_gained']]
        loc['cum_sum'] = loc['p-value'].cumsum()

        coverage_results[mth] = list(loc.coverage_gained.values)
        cumul_p_val[mth] = list(loc.cum_sum.values)
    
        
    for j,(key,values) in enumerate(coverage_results.items()):
        cov_all_results.loc[len(cov_all_results)]=[0.02,0.,methods[j],' '*(len(methods)-j)]
        
        for (cov,p) in zip(values,cumul_p_val[key]):
            cov_all_results.loc[len(cov_all_results)]=[cov,p,methods[j],' '*(len(methods)-j)]
    
    cov_all_results['legend_method'] = cov_all_results['Abv_method']+cov_all_results['Method']
    
    return cov_all_results

def get_coverage_samples_results(res_global_stats, res_global, num_hypo, threshold=-1):
    cov_ahypo = pd.DataFrame(columns=['Coverage','Sample','Method','Abv_method'])
    
    methods = list(res_global_stats.algorithm.unique())
    
    # Get for each method (in each sample) the coverage
    for num in num_hypo:
        for j,mth in enumerate(methods):

            loc = res_global[ (res_global.algorithm==mth) &(res_global['sample']==num*100) & (res_global.top==threshold)]

            cov_ahypo.loc[len(cov_ahypo)] = [loc.coverage_gained.max(),f'{int(num*100)}%',methods[j], ' '*(len(methods)-j)]
    
    cov_ahypo['legend_method'] = cov_ahypo['Abv_method']+cov_ahypo['Method']
    
    return cov_ahypo

def get_time_samples_results(res_global_stats, num_hypo, threshold=-1):
    time_ = pd.DataFrame(columns=['Time','Sample','Method','Abv_method'])
    
    methods = list(res_global_stats.algorithm.unique())
    
    # Get for each method (in each sample) response time
    for num in num_hypo:
        for j,mth in enumerate(methods):
            
            loc = res_global_stats[ (res_global_stats.algorithm==mth) &(res_global_stats['sample']==num*100) \
            & (res_global_stats.top==threshold)]

            time_.loc[len(time_)] = [loc.p_value_adjustement_time.values[0],f'{int(num*100)}%',methods[j],' '*(len(methods)-j)]
    
    time_['legend_method'] = time_['Abv_method']+time_['Method']
    
    return time_

def get_time_n_results(res_global_stats, num_hypo=100):
    time_n = pd.DataFrame(columns=['Time','n','Method','Abv_method'])
    
    methods = list(res_global_stats.algorithm.unique())
    
    # Get for each method and each number of results the response time
    for num in res_global_stats.top.unique():
        for j,mth in enumerate(methods):
            
            loc = res_global_stats[ (res_global_stats.algorithm==mth) &(res_global_stats['sample']==num_hypo)\
            & (res_global_stats.top==num)]

            time_n.loc[len(time_n)] = [loc.p_value_adjustement_time.values[0],f'{num}',methods[j],' '*(len(methods)-j)]
    
    time_n['legend_method'] = time_n['Abv_method']+time_n['Method']
    return time_n