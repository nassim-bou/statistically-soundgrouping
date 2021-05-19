import pandas as pd

d = {0:'Traditional',1:'Coverage_Two_Sides',2:'Coverage_Side_1',3:'Coverage_Side_2', 4:'Coverage_alpha_investing'}

def get_results_csv(dict_stat, dict_res, threshold_n, num_hypo):
    res_global = pd.DataFrame(columns=['group1','group2','coverage_gained_1','coverage_gained_2','p-value','threshold','method','adjustment','num_max_hypo'])

    res_global_stats = pd.DataFrame(columns=['num_max_hypo','threshold','method','adjustment','Candidates','Hypothesis','Without_Adjust',\
    'grouping_time','Best_pairs','Min_p_value', 'Max_p_value','Sum_p_value','Cov_period_1','Cov_period_2','Cov_total','p_value_adjustement_time'])

    adju = {0:'Benjamini',1:'Bonferroni',2:'Benjamini',3:'Bonferroni',4:'Benjamini',5:'Bonferroni',6:'Benjamini',7:'Bonferroni',\
    8:'gamma_20',9:'gamma_50',10:'gamma_100',11:'gamma_200'}

    for num in num_hypo:
        res_ = dict_res[num]
        stats_ = dict_stat[num]

        for top in threshold_n:
            for j,base in enumerate([0,1,2,3,4]):
                local = res_[top][d[base]]

                for i in range(len(local)):
                    res_local = local[i]
                    res_local['threshold'] = top
                    res_local['method'] = d[base]
                    res_local['adjustment'] = adju[2*j+i]
                    res_local['num_max_hypo'] = num
                    res_global = res_global.append(res_local)

                local = stats_[top][j]
                
                if 'Candidates' in local.columns:
                    shared_col = local[['Candidates','Independant_Candidates','Without_Adjust','grouping_time']]
                    local = local.drop(columns=['Candidates','Independant_Candidates','Without_Adjust','grouping_time'])
                    shared_col = shared_col.values.tolist()[0]
                else:
                    shared_col = [0,0,0,0]

                local = local.iloc[:,3:]
                len_ = int(len(local.T)/8)

                for i in range(len_):
                    res_local = local.iloc[:,i*8:(i+1)*8]

                    res_global_stats.loc[len(res_global_stats)] = [num,top,d[base],adju[2*j+i]] + shared_col +\
                    res_local.values.tolist()[0]
                    
    res_global['pairs'] = res_global['group1'] + ' ' + res_global['group2']
    res_global['coverage_gained'] = (res_global['coverage_gained_1'] + res_global['coverage_gained_2'])/2
    
    return res_global_stats, res_global

def get_fdr_power_results(res_global_stats, res_global, num_hypo, threshold=-1):
    
    fdr_power_all_results = pd.DataFrame(columns=['Metric','Sample','Value','Method','Abv_method'])
    lll = {0:'COVER_G_BN',1:'COVER_G_BY',2:'COVER_⍺_20', 3:'COVER_⍺_50',4:'COVER_⍺_100', 5:'COVER_⍺_200',6:'TRAD_BY',
          7:'Perfect'}
    
    fdr_result = dict()
    power_result = dict()
    nb_results = dict()

    for num in num_hypo:
        ground_truth = res_global[ (res_global.method == 'Traditional') & (res_global.adjustment == 'Bonferroni')\
        & (res_global.num_max_hypo == num) & (res_global.threshold == -1)]

        fdr = []
        power = []
        nb_resu = []

        for mth,adju in [('Coverage_Two_Sides','Bonferroni'),('Coverage_Two_Sides','Benjamini'),
                         ('Coverage_alpha_investing','gamma_20'),('Coverage_alpha_investing','gamma_50'),
                         ('Coverage_alpha_investing','gamma_100'),('Coverage_alpha_investing','gamma_200'),
                         ('Traditional','Benjamini'),('Traditional','Bonferroni')]:

            loc = res_global[ (res_global.method==mth) & (res_global.adjustment==adju) & (res_global.num_max_hypo==num)\
            & (res_global.threshold==threshold)]

            loc_2 = res_global_stats[ (res_global_stats.method==mth) & (res_global_stats.adjustment==adju) &\
            (res_global_stats.num_max_hypo==num) & (res_global_stats.threshold==threshold)]

            power.append( len(set(ground_truth.pairs.unique()).intersection(set(loc.pairs.unique()))) /len(ground_truth.pairs.unique()) )
            fdr.append( len(set(loc.pairs.unique()) - set(ground_truth.pairs.unique())) /len(loc.pairs.unique()) )
            nb_resu.append( loc_2.Best_pairs.values[0] )

        power_result[num] = power
        fdr_result[num] = fdr
        nb_results[num] = nb_resu
    
    for key,values in fdr_result.items():
        for j,v in enumerate(values):
            fdr_power_all_results.loc[len(fdr_power_all_results)]=['fdr',f'{int(key)}%',v,lll[j],' '*j]

    for key,values in power_result.items():
        for j,v in enumerate(values):
            fdr_power_all_results.loc[len(fdr_power_all_results)]=['power',f'{int(key)}%',v,lll[j],' '*j]

    for key,values in nb_results.items():
        for j,v in enumerate(values):
            fdr_power_all_results.loc[len(fdr_power_all_results)]=['nb_resu',f'{int(key)}%',v,lll[j],' '*j]
            
    return fdr_power_all_results

def get_coverage_p_values_results(res_global, num_hypo, threshold=-1):
    cov_all_results = pd.DataFrame(columns=['Coverage','p_values','Method'])
    lll_2 = {0:'TRAD_BY',1:'COVER_G_BY',2:'COVER_⍺_20',3:'COVER_⍺_50',4:'COVER_⍺_200'}
    
    cumul_p_val = dict()
    coverage_results = dict()
    
    if isinstance(num_hypo,int):
        num = num_hypo
    else:
        num = list(num_hypo)[-1]
    
    tot_cov = []
    sum_p = []

    for mth,adju in [('Traditional','Benjamini'),('Coverage_Two_Sides','Benjamini'),
                     ('Coverage_alpha_investing','gamma_20'),\
                     ('Coverage_alpha_investing','gamma_50'),('Coverage_alpha_investing','gamma_200')]:
        
        loc = res_global[ (res_global.method==mth) & (res_global.adjustment==adju) & (res_global.num_max_hypo==num)\
        & (res_global.threshold==threshold)]
        
        loc = loc[['p-value','pairs','coverage_gained']]
        loc['cum_sum'] = loc['p-value'].cumsum()

        coverage_results[mth+'_'+adju] = list(loc.coverage_gained.values)
        cumul_p_val[mth+'_'+adju] = list(loc.cum_sum.values)
    
        
    for j,(key,values) in enumerate(coverage_results.items()):
        cov_all_results.loc[len(cov_all_results)]=[0.02,0.,lll_2[j]]
        
        for (cov,p) in zip(values,cumul_p_val[key]):
            cov_all_results.loc[len(cov_all_results)]=[cov,p,lll_2[j]]
    
    return cov_all_results

def get_coverage_samples_results(res_global, num_hypo, threshold=-1):
    cov_ahypo = pd.DataFrame(columns=['Coverage','Sample','Method'])
    lll = {0:'COVER_G_BN',1:'COVER_G_BY',2:'COVER_⍺_20', 3:'COVER_⍺_50',4:'COVER_⍺_100', 5:'COVER_⍺_200',
           6:'TRAD_BN',7:'TRAD_BY'}
    
    for num in num_hypo:
        for j,(mth,adju) in enumerate([('Coverage_Two_Sides','Bonferroni'),('Coverage_Two_Sides','Benjamini'),
                                        ('Coverage_alpha_investing','gamma_20'),\
                                        ('Coverage_alpha_investing','gamma_50'),
                                        ('Coverage_alpha_investing','gamma_100'),
                                        ('Coverage_alpha_investing','gamma_200'),
                                        ('Traditional','Bonferroni'),('Traditional','Benjamini')]):

            loc = res_global[ (res_global.method==mth) & (res_global.adjustment==adju) & (res_global.num_max_hypo==num)\
            & (res_global.threshold==threshold) ]

            cov_ahypo.loc[len(cov_ahypo)] = [loc.coverage_gained.max(),f'{num}%',lll[j]]
            
    return cov_ahypo

def get_time_samples_results(res_global_stats, num_hypo, threshold=-1):
    time_ = pd.DataFrame(columns=['Time','Sample','Method'])
    lll = {0:'COVER_G_BN',1:'COVER_G_BY',2:'COVER_⍺_20', 3:'COVER_⍺_50',4:'COVER_⍺_100', 5:'COVER_⍺_200',
           6:'TRAD_BN',7:'TRAD_BY'}
    
    for num in num_hypo:
        for j,(mth,adju) in enumerate([('Coverage_Two_Sides','Bonferroni'),('Coverage_Two_Sides','Benjamini'),
                                       ('Coverage_alpha_investing','gamma_20'),
                                       ('Coverage_alpha_investing','gamma_50'),
                                       ('Coverage_alpha_investing','gamma_100'),
                                       ('Coverage_alpha_investing','gamma_200'),
                                       ('Traditional','Bonferroni'),('Traditional','Benjamini')]):

            loc = res_global_stats[ (res_global_stats.method==mth) & (res_global_stats.adjustment==adju) &\
            (res_global_stats.num_max_hypo==num) & (res_global_stats.threshold==threshold)]

            time_.loc[len(time_)] = [loc.p_value_adjustement_time.values[0],f'{num}%',lll[j]]
    
    return time_

def get_time_n_results(res_global_stats, num_hypo=100):
    time_n = pd.DataFrame(columns=['Time','n','Method'])
    lll = {0:'COVER_G_BN',1:'COVER_G_BY',2:'COVER_⍺_20', 3:'COVER_⍺_50',4:'COVER_⍺_100', 5:'COVER_⍺_200',
           6:'TRAD_BN',7:'TRAD_BY'}
    
    for num in res_global_stats.threshold.unique():
        for j,(mth,adju) in enumerate([('Coverage_Two_Sides','Bonferroni'),
                                       ('Coverage_Two_Sides','Benjamini'),
                                       ('Coverage_alpha_investing','gamma_20'),\
                                       ('Coverage_alpha_investing','gamma_50'),
                                       ('Coverage_alpha_investing','gamma_100'),
                                       ('Coverage_alpha_investing','gamma_200'),
                                       ('Traditional','Bonferroni'),('Traditional','Benjamini')]):

            loc = res_global_stats[ (res_global_stats.method==mth) & (res_global_stats.adjustment==adju) & 
                                   (res_global_stats.threshold==num) & (res_global_stats.num_max_hypo == num_hypo)]

            time_n.loc[len(time_n)] = [loc.p_value_adjustement_time.values[0],f'{num}',lll[j]]
    
    return time_n