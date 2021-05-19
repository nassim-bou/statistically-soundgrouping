import pandas as pd
from itertools import product,chain
from multiprocessing import Pool
import random
import time
import math

import config
from config import THRESHOLD_INDEPENDENT_TEST
from source_code.core.hypothesis_evaluation.test_evaluation import StatisticalTestHandler
from source_code.core.pivot_handler import PeriodHandler
from source_code.core.group_handler import GroupHandler
from source_code.models.models import get_data
from source_code.utils.tools import add_index, name_groups
import warnings

idx2method = {0:'Traditional',1:'Coverage_Two_Sides',2:'Coverage_Side_1',3:'Coverage_Side_2',4:'Coverage_alpha_investing'}

pd.options.mode.chained_assignment = None
warnings.simplefilter('ignore', category=UserWarning)

def baseline_test_groups(top_n, dataset, approach, period_type, periods_start, period_arg, grp_type, grp_val ,agg_type, dimension, test_arg, support, alpha, verbose=False):
    """
    top_n : integer - Maximum of returned results (n)
    dataset : string - Name of the Dataset
    approach : list of Integer - p value / coverage based / Alpha investinf
    period_type : string - Time_Based or Point_Based time intervals
    periods_start : datetime (or integer) - Datetimes of starting periods (Number of periods in case of point_based intervals)
    period_arg : list of integer - [Number of months,Number of days] (Number of points in the case of point_based intervals)
    grp_type : list - features for clustering
    grp_val : list - values of clustering features
    agg_type : string - Aggregation function
    dimension : string - variable used by aggregate function
    test_arg : Pair - [Type of test (one sample, two samples ...), Value for One Sample]
    support : integer - Minimum number of points in a group
    alpha : float - Threshold of p-value
    verbose : Boolean - If True, print of steps
    """
    one_sample = False

    group_handler = GroupHandler()
    period_handler = PeriodHandler()
    test_handler = StatisticalTestHandler()

    df, feature2values = get_data(dataset=dataset)

    if verbose==True:
        print('Reading data Done')

    #Indexing the users and items with unique integer index
    df, cust_index2id, article_index2id = add_index(df)

    #Split into periods
    start_time = time.time()
    df = period_handler.period(df, test_arg[0], period_type, periods_start, period_arg)

    if verbose==True:
        print('Splitting data into periods Done')
        st = ''
        for i in range(len(df)):
            st = st + f'Number of instances in period {i+1} : {len(df[i])}\n'
        
        print(st)

    if len(grp_type) == 1: #Duplicate item and user features for group creation
        grp_type.append(grp_type[0])
        grp_val.append(grp_val[0])
    
    if agg_type == 'anova':#Duplicate item and user features for group creation in cse of Anova
        if isinstance(periods_start[0],int):
            len__ = periods_start[0] - len(grp_type)
        else:
            len__ = len(periods_start) - len(grp_type)

        while len__ > 0:
            grp_type.append(grp_type[0])
            grp_val.append(grp_val[0])
            len__ = len__ - 1

    #Pre-grouping users based on condition + Create all groups
    df = [group_handler.pre_group(d, grp_type[i], grp_val[i]) for i,d in enumerate(df)]

    for i in range(len(df)): #Missing data in one of the periods
        if len(df[i]) == 0:
            return None, None, None

    groups = [ group_handler.groups(d, grp_type[i]) for i,d in enumerate(df)]
    time_groups = time.time() - start_time

    del df

    if verbose==True:
        len__ = 1
        for g in groups:
            len__ = len__ * len(g)
        
        print( f'All condidates : {len__}' )

    #Keep groups with a minimum support (Number of users)
    groups = [ [df for df in grp if len(df)>support  ] for grp in groups ]

    #dict = Key:Group description - value: dataframe
    nameGrp_2_index = [ {name_groups(df):df for idx,df in enumerate(grp)} for grp in groups ]

    cases = []

    start_time = time.time()

    if test_arg[0]=='Paired': #In case of Paired test
        
        gg = set(nameGrp_2_index[0].keys())

        for gr in nameGrp_2_index[1:]: #Similar groups in the periods
            gg = gg.intersection(set(gr.keys()))

        for case in gg: #Create all cases
            l = []
            for gr in nameGrp_2_index:
                l.append(gr[case])

            cases.append( l )

    paired_time = time.time() - start_time
    #Get the list of users of each group | dict = key: Group description - value: List of users 
    group_2_users = [ {name_groups(df):set(df.cust_id.unique()) for df in grp} for grp in groups ]
    group1_2_users = group_2_users[0]

    group2_2_users = dict()

    if len(groups)>1:
        group2_2_users = group_2_users[1]

    group3_2_users = []
    if len(groups)>2:
        group3_2_users = [i for i in group_2_users[2:]]

    #Get the list of users of each period
    users1 = set()
    for key,value in group1_2_users.items():
        users1 = users1.union(value)
    
    users2 = set()
    if len(groups)>1:
        for key,value in group2_2_users.items():
            users2 = users2.union(value)

    if len(groups) < 2: #If ONE-SAMPLE == TRUE
        groups.append( [test_arg[1]] )
        one_sample = True

    len_e = len(groups[0])
    len_h = len(groups[1])

    if verbose==True:
        if len(groups) > 2:
            len_k = 1
            for g in groups[2:]:
                len_k = len_k * len(g)
        else:
            len_k = 1

        if test_arg[0]=='Paired':
            print(f'Grouping Data Done - All condidates keeping ones above the support : {min(len_e,len_h)}')
        else:
            print(f'Grouping Data Done - All condidates keeping ones above the support : {len_e * len_h * len_k}')

    # Cartesian product +  Evaluate all cases
    if test_arg[0]!='Paired':
        start_time = time.time()

        cases = list(product(*groups)) #Cartesian product

        time_generate_all_cases = time.time() - start_time
    else:
        start_time = time.time()

        pool = Pool()      
        cases = pool.map(identique_pair, cases)
        pool.close()

        cases = [case for case in cases if len(case)>0]

        time_generate_all_cases = time.time() - start_time + paired_time
    
    nb_cases = len(cases)
    len_groups = len(groups)
    
    #Get independant and normally distributed samples
    result = test_handler.evaluate_mult(agg_type, cases, test_arg[0])

    nb__ = len(result)

    if len_groups > 2:
        for i in range(len_groups):
            result = result[result[f"chi-squared test {i}"] > THRESHOLD_INDEPENDENT_TEST]
    else:
        result = result[result["chi-squared test"] > THRESHOLD_INDEPENDENT_TEST]

    if verbose==True:
        print( 'All independant cases : ',nb__ )
    
    col = [f'grp{i+1}' for i in range(len_groups)]
    cases = []

    for grps in result[col].itertuples(): #Recreate all independant cases
        grps = grps[1:]
        if len(nameGrp_2_index)==1: #One-Sample test
            cases.append( (nameGrp_2_index[0][grps[0]],grps[1]) )
        else:
            tupl = tuple()
            for i,j in enumerate(grps):
                tupl = tupl + (nameGrp_2_index[i][j],)
            cases.append( tupl )

    numHypo_2_stats = dict()
    numHypo_2_results = dict()

    columns_stat = ['Candidates','Independant_Candidates','Without_Adjust','Min_p_value','Max_p_value','Sum_p_value',
    'Best_pairs_BY','Min_p_value_BY','Max_p_value_BY','Sum_p_value_BY','Cov_period_1_BY','Cov_period_2_BY','Cov_total_BY','p_value_adjustement_BY_time',
    'Best_pairs_B','Min_p_value_B','Max_p_value_B','Sum_p_value_B','Cov_period_1_B','Cov_period_2_B','Cov_total_B','p_value_adjustement_B_time','grouping_time']

    columns_stat_Alpha_investing = ['Candidates','Independant_Candidates','Without_Adjust','Min_p_value','Max_p_value','Sum_p_value'] +\
    ['Best_pairs_20','Min_p_value_20','Max_p_value_20','Sum_p_value_20','Cov_period_1_20','Cov_period_2_20','Cov_total_20','p_value_adjustement_time_20'] +\
    ['Best_pairs_50','Min_p_value_50','Max_p_value_50','Sum_p_value_50','Cov_period_1_50','Cov_period_2_50','Cov_total_50','p_value_adjustement_time_50'] +\
    ['Best_pairs_100','Min_p_value_100','Max_p_value_100','Sum_p_value_100','Cov_period_1_100','Cov_period_2_100','Cov_total_100','p_value_adjustement_time_100']+\
    ['Best_pairs_200','Min_p_value_200','Max_p_value_200','Sum_p_value_200','Cov_period_1_200','Cov_period_2_200','Cov_total_200','p_value_adjustement_time_200']+\
    ['grouping_time']

    #for num_hyp in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]: #Samples
    for num_hyp in [1]: #Samples
        leng_sample = int(num_hyp*nb__)

        if verbose==True:
            print(num_hyp*100,'% ',leng_sample)

        if num_hyp < 1:
            sampled_cases = random.sample(cases, leng_sample)
            random.shuffle(sampled_cases)
        else:
            sampled_cases = cases
            random.shuffle(sampled_cases)

        start_time = time.time() #Get the real time of computing all p-values

        result = test_handler.evaluate(agg_type, sampled_cases, test_arg[0]) #Columns : name_grp1, members_grp1, size_grp1, name_grp2, members-grp2, size_grp2, p_value, chi2_test
        result_alpha = result[col].copy()

        time_process_all_cases = time.time() - start_time

        if verbose==True:
            print('p-values and chi2-test Done')

        all_candidates = len(result)
        result = result[result["p-value"] < alpha]

        nb_under_p_value = len(result)

        if 'grp1_members' in result.columns:
            columns = [f'grp{i+1}_members' for i in range(len_groups)]
            result.drop(columns=columns,inplace=True)

        all_stats = None
        all_results = None

        if len(result)>0:
            all_stats = dict()
            all_results = dict()

            min_p_value = result["p-value"].min()
            max_p_value = result["p-value"].max()

            if len_groups > 2: #Anova
                chis_2 = [f'chi-squared test {i}' for i in range(len_groups)]
                result1 = result.drop(chis_2, axis=1).copy()
            else:
                result1 = result.drop(["chi-squared test"], axis=1).copy()

            del result
            
            for top in top_n: # Thresholds (n)

                results = dict()
                stats = []

                if verbose==True:
                    print('n = ',top)

                for appr in approach: # Methods
                    res = []
                    time_exec = []

                    if appr == 4: #Alpha investing support
                        methods = [20,50,100,200] #Gammas
                        process_time = 0

                        for mtd in [20,50,100,200] :

                            if 'coverage' in result_alpha.columns:
                                result_alpha.drop(columns=['coverage','cov_1','cov_2'], inplace=True)

                            start_time = time.time()

                            res.append(alpha_investing(top, result_alpha, nameGrp_2_index, agg_type, test_arg[0], group1_2_users, group2_2_users, users1, users2,\
                             alpha=alpha, method=mtd))

                            time_exec.append(time.time() - start_time)
                            
                            if verbose==True:
                                print(f'Alpha investing : {mtd} - time processing = {time_exec[-1]} s')
                    else:
                        methods = ["fdr_by", "fdr_b"]
                        process_time = time_process_all_cases

                        for mtd in ["fdr_by", "fdr_b"]:
                            start_time = time.time()

                            res.append(trad_and_cover_g(top, appr, result1, group1_2_users, group2_2_users, users1, users2, alpha=alpha, method=mtd))

                            time_exec.append(time.time() - start_time)

                            if verbose==True:
                                print(f'Method : {appr} - Adjust : {mtd} - time processing = {time_exec[-1]+time_process_all_cases} s')
                            
                    stat = [int(num_hyp*nb_cases), all_candidates, nb_under_p_value,\
                    result1['p-value'].min(), result1['p-value'].max(), result1['p-value'].sum()]

                    if one_sample == False:
                        for i in range(len(methods)):
                            stat = stat + [len(res[i]),res[i]['p-value'].min(), res[i]['p-value'].max(), res[i]['p-value'].sum() ,\
                            res[i]['coverage_gained_1'].max(), res[i]['coverage_gained_2'].max(),\
                            (res[i]['coverage_gained_1'].max()+res[i]['coverage_gained_2'].max())/2,time_exec[i]+process_time]
                    else:
                        for i in range(len(methods)):
                            stat = stat + [len(res[i]),res[i]['p-value'].min(), res[i]['p-value'].max(), res[i]['p-value'].sum() ,\
                            res[i]['coverage_gained_1'].max(), res[i]['coverage_gained_2'].max(),\
                            res[i]['coverage_gained_1'].max(),time_exec[i]+process_time]

                    if appr < 4 :
                        stats.append(pd.DataFrame(stat+[time_groups], index=columns_stat ).T)
                    else:
                        stats.append(pd.DataFrame(stat+[time_groups], index=columns_stat_Alpha_investing ).T)
                    
                    results[ idx2method[appr] ] = res
                
                all_stats[top] = stats
                all_results[top] = results
        
        numHypo_2_stats[num_hyp*100] = all_stats
        numHypo_2_results[num_hyp*100] = all_results

    return numHypo_2_stats, numHypo_2_results, result1

def trad_and_cover_g(top_n, approach, result, group1_2_users, group2_2_users, users1, users2, alpha=0.05, method="fdr_by"):
    """
    approach :
    0 = Traditionnal
    1 = Sum of Coverage
    2 = Coverage Grp 1
    3 = Coverage Grp 2
    """
    top = top_n

    if top == -1:
        top = 1000000000
    
    coverage_grp1 = 0
    coverage_grp2 = 0

    users_1 = users1
    users_2 = users2

    len_u1 = len(users1)
    len_u2 = len(users2)

    one_sample = False

    if len_u2 == 0:
        one_sample = True
        len_u2 = 1

    res = []

    if result.grp2.dtype == float:
        result.grp2 = result.grp2.astype(str)

    m = result.shape[0]
    nb_columns = len(result.columns)

    grp1_duplicated = set(result.grp1.unique())

    if len(group2_2_users)==0:
        grp2_duplicated = set()
        group2_2_users[ list(result.grp2.unique())[0] ]={}
    else:
        grp2_duplicated = set(result.grp2.unique())

    result = result.sort_values("p-value").reset_index(drop=True)

    if approach > 0: #Cover_G
        d = result[['grp1','grp2']]
        d['key'] = d.apply(lambda x : x[0]+' '+x[1],axis=1)
        d.drop(columns=['grp1','grp2'],inplace=True)
        d['val'] = d.index
        d = d.set_index('key').T.to_dict('list')

    for n in range(len(result)):
        if approach > 0: #Cover_G
            if approach == 2:
                result['coverage'] = result.grp1.apply(lambda x : len(users_1.intersection( group1_2_users[x] )) )
            if approach == 3:
                result['coverage'] = result.grp2.apply(lambda x : len(users_2.intersection( group2_2_users[x] )) )
            if approach == 1 :
                result['cov_1'] = result.grp1.apply(lambda x : len(users_1.intersection( group1_2_users[x] )) )
                result['cov_2'] = result.grp2.apply(lambda x : len(users_2.intersection( group2_2_users[x] )) )
                result['coverage'] = result['cov_1']+result['cov_2']
                result.drop(columns=['cov_1','cov_2'],inplace=True)

            result = result[result.coverage > 0]

            if len(result)>0:
                best = result['coverage'].idxmax()
            
            result.drop(columns=['coverage'],inplace=True)
            
        if (len(result)==0) or (top==0):
            break

        if approach > 0: #Cover_G
            if nb_columns > 5: #ANOVA CASE
                grp1 = result.loc[[best]].values.tolist()[0]
                p = grp1[-1]
                grp1 = grp1[:-1]
                gr = [ grp1[jj] for jj in range(0,len(grp1),2)]
                grp1, grp2 = gr[0],gr[1]

            else:
                grp1, grp1_size, grp2, grp2_size, p = result.loc[[best]].values.tolist()[0]

            result = result.drop(best)
            pos = d[grp1+' '+grp2][0]
        else: #TRAD
            if nb_columns > 5: #ANOVA CASE
                grp1 = result.head(1).values.tolist()[0]
                p = grp1[-1]
                grp1 = grp1[:-1]
                gr = [ grp1[jj] for jj in range(0,len(grp1),2)]
                grp1, grp2 = gr[0],gr[1]
            else:
                grp1, grp1_size, grp2, grp2_size, p = result.head(1).values.tolist()[0]

            result = result.iloc[1:]
            pos = n

        limit = compute_limit(alpha, pos + 1, m, method)

        if p >= limit:
            if approach == 0:
                break
            else:
                continue
        
        if grp1 in grp1_duplicated:
            grp1_duplicated.remove(grp1)

            inter = users_1.intersection( group1_2_users[grp1] )
            users_1 = {i for i in users_1 if i not in inter} 
            coverage_grp1 += len( inter )

        if grp2 in grp2_duplicated:
            grp2_duplicated.remove(grp2)

            inter = users_2.intersection( group2_2_users[grp2] )
            users_2 = {i for i in users_2 if i not in inter} 
            coverage_grp2 += len( inter )

        res.append( (grp1, grp2, coverage_grp1/len_u1, coverage_grp2/len_u2, p) )

        top = top-1

    df_results = pd.DataFrame(res, columns=["group1","group2","coverage_gained_1","coverage_gained_2","p-value"])

    return df_results

def alpha_investing(top_n, result, nameGrp_2_index, test_type, test_arg, group1_2_users, group2_2_users, users1, users2, alpha=0.05, method="fdr_by"):        
    
    test_handler = StatisticalTestHandler()

    top = top_n

    if top == -1:
        top = 1000000000
    
    coverage_grp1 = 0
    coverage_grp2 = 0

    users_1 = users1
    users_2 = users2

    len_u1 = len(users1)
    len_u2 = len(users2)

    one_sample = False

    if len_u2 == 0:
        one_sample = True
        len_u2 = 1

    res = []

    if result.grp2.dtype == float:
        result.grp2 = result.grp2.astype(str)

    m = result.shape[0]
    nb_columns = len(result.columns)

    grp1_duplicated = set(result.grp1.unique())

    if len(group2_2_users)==0:
        grp2_duplicated = set()
        group2_2_users[ list(result.grp2.unique())[0] ]={}
    else:
        grp2_duplicated = set(result.grp2.unique())

    w0 = (1-alpha)*alpha
    w_next = w0

    if isinstance(method,str):
        gamma = 100
    else:
        gamma = method
    
    alpha_0 = w0 / (gamma+w0)

    cov_calcul = True
    n_refus = 0

    while (w_next > 0) and (top > 0):
        if cov_calcul == True:
            result['cov_1'] = result.grp1.apply(lambda x : len(users_1.intersection( group1_2_users[x] )) )
            result['cov_2'] = result.grp2.apply(lambda x : len(users_2.intersection( group2_2_users[x] )) )
            result['coverage'] = result['cov_1']+result['cov_2']
            result = result[result.coverage > 0]

        cov_calcul = False

        if len(result) > 0 :
            best = result['coverage'].idxmax()
        else:
            result.drop(columns=['cov_1','cov_2','coverage'],inplace=True)
            break

        case = []

        if nb_columns > 2: #ANOVA CASE
            grp1 = result.loc[[best]].values.tolist()[0]
            cov1,cov2 = grp1[-3], grp1[-2]
            grp1 = grp1[:-3]
            case = []
            for idx_grp, grp_i in enumerate(grp1):
                case.append(nameGrp_2_index[idx_grp][grp_i])

            grp2 = grp1[1]
            grp1 = grp1[0]
        else:
            grp1, grp2, cov1, cov2, cov_tot  = result.loc[[best]].values.tolist()[0]
            case = [ nameGrp_2_index[0][grp1], nameGrp_2_index[1][grp2] ]

        result = result.drop(best)

        p = test_handler.evaluate_one(test_type, case, test_arg)

        if len(p)==0:
            continue
        
        p = p['p-value'].values[0]

        cov_j = (cov1/len_u1) + (cov2/len_u2)
        if one_sample == False:
            cov_j = cov_j / 2

        alpha_j = alpha_0 * math.sqrt(cov_j)
        err = w_next - ( alpha_j/(1-alpha_j) )

        if err >= 0:
            if p <= alpha_j:
                w_next = w_next + alpha_j
                n_refus = 0
            else:
                w_next = err
                n_refus = n_refus + 1
                continue
        else:
            continue

        cov_calcul = True

        if grp1 in grp1_duplicated:
            grp1_duplicated.remove(grp1)
            inter = users_1.intersection( group1_2_users[grp1] )
            users_1 = {i for i in users_1 if i not in inter} 

            coverage_grp1 += len(inter)

        if grp2 in grp2_duplicated:
            grp2_duplicated.remove(grp2)
            inter = users_2.intersection( group2_2_users[grp2] )
            users_2 = {i for i in users_2 if i not in inter} 

            coverage_grp2 += len(inter)

        res.append( (grp1, grp2, coverage_grp1/len_u1, coverage_grp2/len_u2, p , w_next-alpha_j, alpha_j, n_refus) )
        top = top-1    

    df_results = pd.DataFrame(res, columns=["group1","group2","coverage_gained_1","coverage_gained_2","p-value","wealth","alpha_j","nb_refus_before"])

    return df_results

def compute_limit(alpha, n, m, method="fdr_by"):
    if method == "fdr_by":
        return alpha * n / (m * sum(1 / ii for ii in range(1, m+1)))
    if method == "fdr_b":
        return alpha / m

def identique_pair(x):
    i = x[0]
    j = x[1]

    i = set(i.cust_id.unique())
    j = set(j.cust_id.unique())

    if i == j:
        return x
    else:
        return []
