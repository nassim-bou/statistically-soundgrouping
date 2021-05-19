import pandas as pd
import numpy as np
import statistics
import itertools
from functools import partial
from multiprocessing import Pool
import sys

from scipy import stats
from scipy.stats import chi2_contingency, kstest, f_oneway
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.weightstats import ztest

from config import THRESHOLD_NORMAL_DIST,THRESHOLD_INDEPENDENT_TEST
from source_code.utils.tools import name_groups

class StatisticalTestHandler:

    # EVALUATION OF MANY CASES USING MULTI PROCESSING
    def evaluate_mult(self, test_type, cases, test_arg, segmentation_type = None, segmentation_arg = None):
        if test_type == "mean":
            return self.mean_evaluation_mult(cases, test_arg, segmentation_type, segmentation_arg)
        elif test_type == "distribution":
            return self.distribution_evaluation_mult(cases, test_arg, segmentation_type, segmentation_arg)
        elif test_type == "variance":
            return self.variance_evaluation_mult(cases, test_arg, segmentation_type, segmentation_arg)
        elif test_type == "anova":
            return self.anova_evaluation_mult(cases, test_arg, segmentation_type, segmentation_arg)
        
        raise Exception("test type not implemented yet ")

    def distribution_evaluation_mult(self, cases, test_arg, segmentation_type, segmentation_arg):
        if test_arg == "One-Sample":
            test_func = self.one_sample_dist_test
        elif test_arg == "Two-Samples":
            test_func = self.two_sample_dist_test
        else:
            raise NotImplementedError("Distribution test type not implemented")
        self.test_arg = test_arg

        f = partial(compute_test_parallel, test_func=test_func, seg_type=segmentation_type, seg_arg=segmentation_arg)

        pool = Pool()
        res = pool.map(f, cases)
        pool.close()
        
        res = pd.DataFrame(res, columns=["grp1", "grp1_members", "grp1_size", "grp2", "grp2_members", "grp2_size", "p-value", "chi-squared test"])
        res = res[~res["p-value"].isna()]

        return res

    def mean_evaluation_mult(self, cases, test_arg, segmentation_type, segmentation_arg):
        if test_arg == "One-Sample":
            test_func = self.one_sample_mean_test
        elif test_arg == "Two-Samples":
            test_func = self.two_sample_mean_test
        elif test_arg == "Paired":
            test_func = self.paired_mean_test
        elif test_arg[0] == "z_test":
            test_func = self.z_test
        else:
            raise NotImplementedError("Mean test type not implemented")
        self.test_arg = test_arg

        f = partial(compute_test_parallel, test_func=test_func, seg_type=segmentation_type, seg_arg=segmentation_arg)  

        pool = Pool()      
        res = pool.map(f, cases)
        pool.close()

        res = pd.DataFrame(res, columns=["grp1", "grp1_members", "grp1_size", "grp2", "grp2_members", "grp2_size", "p-value", "chi-squared test"])
        res = res[~res["p-value"].isna()]

        return res

    def variance_evaluation_mult(self, cases, test_arg, segmentation_type, segmentation_arg):
        if test_arg == "Two-Samples":
            test_func = self.F_test
        if test_arg == "One-Sample":
            test_func = self.one_sample_variance_test
        else:
            raise NotImplementedError("Other Variance test is not implemented")

        self.test_arg = test_arg

        f = partial(compute_test_parallel, test_func=test_func, seg_type=segmentation_type, seg_arg=segmentation_arg)

        pool = Pool()
        res = pool.map(f, cases)
        pool.close()

        res = pd.DataFrame(res, columns=["grp1", "grp1_members", "grp1_size", "grp2", "grp2_members", "grp2_size", "p-value", "chi-squared test"])
        res = res[~res["p-value"].isna()]

        return res

    def anova_evaluation_mult(self, cases, test_arg, segmentation_type, segmentation_arg):
        f = partial(compute_test_parallel, test_func=self.anova, seg_type=segmentation_type, seg_arg=segmentation_arg)

        pool = Pool()
        res = pool.map(f, cases)
        pool.close()
            
        len_groups = len(cases[0])        
        columns = []

        for i in range(len_groups):
            columns = columns + [f"grp{i+1}", f"grp{i+1}_members", f"grp{i+1}_size"]

        len_groups = len(res[0]) - len(columns) - 1
        columns = columns + [ "p-value"] + [ f"chi-squared test {i}" for i in range(len_groups)]

        res = pd.DataFrame(res, columns=columns)
        res = res[~res["p-value"].isna()]

        return res

    # EVALUATION OF MANY CASES (WITHOUT MULTI PROCESSING)
    def evaluate(self, test_type, cases, test_arg, segmentation_type = None, segmentation_arg = None):
        if test_type == "mean":
            return self.mean_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        elif test_type == "distribution":
            return self.distribution_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        elif test_type == "variance":
            return self.variance_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        elif test_type == "anova":
            return self.anova_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        
        raise Exception("test type not implemented yet ")

    def distribution_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):
        if test_arg == "One-Sample":
            test_func = self.one_sample_dist_test
        elif test_arg == "Two-Samples":
            test_func = self.two_sample_dist_test
        else:
            raise NotImplementedError("Distribution test type not implemented")
        self.test_arg = test_arg


        f = partial(compute_test_parallel, test_func=test_func, seg_type=segmentation_type, seg_arg=segmentation_arg)
        res = []
        for case in cases:
            res.append( f(case) )
        
        res = pd.DataFrame(res, columns=["grp1", "grp1_members", "grp1_size", "grp2", "grp2_members", "grp2_size", "p-value", "chi-squared test"])
        res = res[~res["p-value"].isna()]

        return res

    def mean_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):
        if test_arg == "One-Sample":
            test_func = self.one_sample_mean_test
        elif test_arg == "Two-Samples":
            test_func = self.two_sample_mean_test
        elif test_arg == "Paired":
            test_func = self.paired_mean_test
        elif test_arg[0] == "z_test":
            test_func = self.z_test
        else:
            raise NotImplementedError("Mean test type not implemented")
        self.test_arg = test_arg

        f = partial(compute_test_parallel, test_func=test_func, seg_type=segmentation_type, seg_arg=segmentation_arg)  

        '''
        pool = Pool()      
        res = pool.map(f, cases)
        pool.close()
        '''
        res = []
        for case in cases:
            res.append( f(case) )

        res = pd.DataFrame(res, columns=["grp1", "grp1_members", "grp1_size", "grp2", "grp2_members", "grp2_size", "p-value", "chi-squared test"])
        res = res[~res["p-value"].isna()]

        return res

    def variance_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):
        if test_arg == "Two-Samples":
            test_func = self.F_test
        if test_arg == "One-Sample":
            test_func = self.one_sample_variance_test
        else:
            raise NotImplementedError("Other Variance test is not implemented")

        self.test_arg = test_arg

        f = partial(compute_test_parallel, test_func=test_func, seg_type=segmentation_type, seg_arg=segmentation_arg)

        res = []
        for case in cases:
            res.append( f(case) )

        res = pd.DataFrame(res, columns=["grp1", "grp1_members", "grp1_size", "grp2", "grp2_members", "grp2_size", "p-value", "chi-squared test"])
        res = res[~res["p-value"].isna()]

        return res

    def anova_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):
        f = partial(compute_test_parallel, test_func=self.anova, seg_type=segmentation_type, seg_arg=segmentation_arg)

        res = []
        for case in cases:
            res.append( f(case) )

        len_groups = len(cases[0])
        
        columns = []
        for i in range(len_groups):
            columns = columns + [f"grp{i+1}", f"grp{i+1}_members", f"grp{i+1}_size"]

        columns = columns + [ "p-value"] + [ f"chi-squared test {i}" for i in range(len_groups)]

        res = pd.DataFrame(res, columns=columns)
        res = res[~res["p-value"].isna()]

        return res

    # EVALUATION OF ONE CASE
    def evaluate_one(self, test_type, cases, test_arg, segmentation_type = None, segmentation_arg = None):
        if test_type == "mean":
            return self.mean_one_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        elif test_type == "distribution":
            return self.distribution_one_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        elif test_type == "variance":
            return self.variance_one_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        elif test_type == "anova":
            return self.anova_one_evaluation(cases, test_arg, segmentation_type, segmentation_arg)
        
        raise Exception("test type not implemented yet ")
 
    def mean_one_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):
        if test_arg == "One-Sample":
            test_func = self.one_sample_mean_test
        elif test_arg == "Two-Samples":
            test_func = self.two_sample_mean_test
        elif test_arg == "Paired":
            test_func = self.paired_mean_test
        elif test_arg[0] == "z_test":
            test_func = self.z_test
        else:
            raise NotImplementedError("Mean test type not implemented")
        self.test_arg = test_arg

        f = partial(compute_test_parallel, test_func=test_func, seg_type=segmentation_type, seg_arg=segmentation_arg)
        res = f(cases)

        res = pd.DataFrame(res, index=["grp1", "grp1_members", "grp1_size", "grp2", "grp2_members", "grp2_size", "p-value", "chi-squared test"]).T
        res = res[~res["p-value"].isna()]
        res = res[res["chi-squared test"] > THRESHOLD_INDEPENDENT_TEST]

        return res

    def distribution_one_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):
        if test_arg == "One-Sample":
            test_func = self.one_sample_dist_test
        elif test_arg == "Two-Samples":
            test_func = self.two_sample_dist_test
        else:
            raise NotImplementedError("Distribution test type not implemented")
        self.test_arg = test_arg

        f = partial(compute_test_parallel, test_func=test_func, seg_type=segmentation_type, seg_arg=segmentation_arg)
        res = f(cases)

        res = pd.DataFrame(res, index=["grp1", "grp1_members", "grp1_size", "grp2", "grp2_members", "grp2_size", "p-value", "chi-squared test"]).T
        res = res[~res["p-value"].isna()]
        res = res[res["chi-squared test"] > THRESHOLD_INDEPENDENT_TEST]

        return res

    def variance_one_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):
        if test_arg == "Two-Samples":
            test_func = self.F_test
        if test_arg == "One-Sample":
            test_func = self.one_sample_variance_test
        else:
            raise NotImplementedError("Other Variance test is not implemented")

        self.test_arg = test_arg

        f = partial(compute_test_parallel, test_func=test_func, seg_type=segmentation_type, seg_arg=segmentation_arg)
        res = f(cases)

        res = pd.DataFrame(res, index=["grp1", "grp1_members", "grp1_size", "grp2", "grp2_members", "grp2_size", "p-value", "chi-squared test"]).T
        res = res[~res["p-value"].isna()]
        res = res[res["chi-squared test"] > THRESHOLD_INDEPENDENT_TEST]

        return res

    def anova_one_evaluation(self, cases, test_arg, segmentation_type, segmentation_arg):

        f = partial(compute_test_parallel, test_func=self.anova, seg_type=segmentation_type, seg_arg=segmentation_arg)
        res = f(cases)

        len_groups = len(cases)
        
        columns = []
        for i in range(len_groups):
            columns = columns + [f"grp{i+1}", f"grp{i+1}_members", f"grp{i+1}_size"]

        columns = columns + [ "p-value"] + [ f"chi-squared test {i}" for i in range(len_groups)]

        l = len(res)
        res = np.array(res).reshape((1,l))

        res = pd.DataFrame(res, columns=columns)
        res = res[~res["p-value"].isna()]

        for i in range(len_groups):
            res = res[res[f"chi-squared test {i}"] > THRESHOLD_INDEPENDENT_TEST]

        return res


    def one_sample_mean_test(self, i, j, segmentation_type):
        values_1 = aggregate_values(i, segmentation_type)
        return stats.ttest_1samp(values_1, j)[1], 1

    def one_sample_dist_test(self, i, j, segmentation_type):
        values_1 = aggregate_values(i, segmentation_type)
        return stats.kstest(values_1, j)[1], 1

    def two_sample_mean_test(self, i, j, segmentation_type):
        values_1 = aggregate_values(i,segmentation_type)
        values_2 = aggregate_values(j, segmentation_type)

        t_normal = self.test_normal_distribution(values_1)
        r_normal = self.test_normal_distribution(values_2)

        if not (t_normal and r_normal):
            values_1 = (values_1 - min(values_1)) / (max(values_1) - min(values_1))
            values_2 = (values_2 - min(values_2)) / (max(values_2) - min(values_2))

        contingency_df = self.generate_contingency_table(values_1, values_2)
        stat, p, dof, expected = chi2_contingency(contingency_df)

        return stats.ttest_ind(values_1, values_2, equal_var=False, alternative='less')[1], p

    def paired_mean_test(self, i, j, segmentation_type):
        values_1 = aggregate_values(i,segmentation_type)
        values_2 = aggregate_values(j, segmentation_type)

        t_normal = self.test_normal_distribution(values_1)
        r_normal = self.test_normal_distribution(values_2)

        if not (t_normal and r_normal):
            values_1 = (values_1 - min(values_1)) / (max(values_1) - min(values_1))
            values_2 = (values_2 - min(values_2)) / (max(values_2) - min(values_2))

        contingency_df = self.generate_contingency_table(values_1, values_2)
        stat, p, dof, expected = chi2_contingency(contingency_df)

        if len(values_1)==len(values_2):
            k = 1
        else:
            if len(values_1) > len(values_2):
                values_1 = values_1[:len(values_2)]
            else:
                values_2 = values_2[:len(values_1)]

        return stats.ttest_rel(values_1, values_2)[1], p

    def two_sample_dist_test(self, i, j, segmentation_type):
        values_1 = aggregate_values(i,segmentation_type)
        values_2 = aggregate_values(j, segmentation_type)

        t_normal = self.test_normal_distribution(values_1)
        r_normal = self.test_normal_distribution(values_2)

        if not (t_normal and r_normal):
            values_1 = (values_1 - min(values_1)) / (max(values_1) - min(values_1))
            values_2 = (values_2 - min(values_2)) / (max(values_2) - min(values_2))

        contingency_df = self.generate_contingency_table(values_1, values_2)
        stat, p, dof, expected = chi2_contingency(contingency_df)

        return kstest(values_1, values_2)[1], p

    def z_test(self, i, j, freq="D"):
        values_1 = aggregate_values(df=i, freq=freq) / self.test_arg[1]
        values_2 = aggregate_values(df=j, freq=freq) / self.test_arg[1]
        t_normal = self.test_normal_distribution(values_1)
        r_normal = self.test_normal_distribution(values_2)

        if not (t_normal and r_normal):
            values_1 = (values_1 - min(values_1)) / (max(values_1) - min(values_1))
            values_2 = (values_2 - min(values_2)) / (max(values_2) - min(values_2))

        contingency_df = self.generate_contingency_table(values_1, values_2)
        stat, p, dof, expected = chi2_contingency(contingency_df)

        return ztest(values_1, values_2)[1], p
    
    def F_test(self, i, j, segmentation_type):
        values_1 = aggregate_values(i,segmentation_type)
        values_2 = aggregate_values(j, segmentation_type)

        t_normal = self.test_normal_distribution(values_1)
        r_normal = self.test_normal_distribution(values_2)

        if not (t_normal and r_normal):
            values_1 = (values_1 - min(values_1)) / (max(values_1) - min(values_1))
            values_2 = (values_2 - min(values_2)) / (max(values_2) - min(values_2))

        contingency_df = self.generate_contingency_table(values_1, values_2)
        stat, p, dof, expected = chi2_contingency(contingency_df)

        F = statistics.variance(values_1) / statistics.variance(values_2)
        df1 = len(values_1)-1
        df2 = len(values_2)-1
        
        return stats.f.sf(F, df1, df2), p

    def one_sample_variance_test(self, i, j, segmentation_type):
        values_1 = aggregate_values(i, segmentation_type)
        df = len(values_1)-1
        chi = df * statistics.variance(values_1) / float(j)
        return stats.chi2.sf(chi, df), 1

    def anova(self,i, segmentation_type):
        values_s = []
        for v in i:
            values_s.append( aggregate_values(v,segmentation_type) )
        
        t_normal = True
        for v in values_s:
            t_normal = t_normal and self.test_normal_distribution(v) 
        
        if not (t_normal):
            for j in range(len(values_s)):
                values_s[j] = (values_s[j] - min(values_s[j])) / (max(values_s[j]) - min(values_s[j]))

        indep_cases = list(itertools.combinations(values_s,2))

        indep_tests = []

        for i in indep_cases:
            contingency_df = self.generate_contingency_table(i[0], i[1])
            stat, p, dof, expected = chi2_contingency(contingency_df)
            indep_tests.append(p)

        return f_oneway(*values_s)[1],*indep_tests

    def test_normal_distribution(self, values_1, threshold_normal_dist=THRESHOLD_NORMAL_DIST):
        return len(values_1) >= 8 and stats.normaltest(values_1)[1] >= threshold_normal_dist

    def generate_contingency_table(self, values_1, values_2):
        data = pd.DataFrame([values_1, values_2]).T.fillna(0).astype(bool)
        return pd.DataFrame([data[0].value_counts(), data[1].value_counts()]).T #####ICI


def aggregate_values(df, segmentation_type, freq="D", col="rating"):
    if segmentation_type is None:
        return df[col].values
    else:
        return df.groupby(pd.Grouper(freq=freq)).agg({"purchase": "sum"})["purchase"].values #####ICI

def compute_test_parallel(args, test_func, seg_type, seg_arg):
    columns = ['cust_id','article_id','rating','purchase', 'transaction_date']

    if len(args) < 3 :
        i = args[0]
        j = args[1]

        i2 = i
        i = i[['cust_id','article_id','rating','purchase', 'transaction_date']]

        if (isinstance(j,int) == False) and (isinstance(j,str) == False) and (isinstance(j,float) == False) :
            j2 = j
            j = j[['cust_id','article_id','rating','purchase', 'transaction_date']]
            set_2_group = set(j.cust_id.unique())
            len_2_group = len(j)
        else:
            j2 = j
            set_2_group = {}
            len_2_group = 0

        return name_groups(i2), set(i.cust_id.unique()), len(i), name_groups(j2), set_2_group, len_2_group, *test_func(i, j, seg_type)
    else: #ANOVA
        periods = args

        periods_2 = periods
        periods = [ i[['cust_id','article_id','rating','purchase', 'transaction_date']] for i in periods ]

        periods_2 = [ (name_groups(i), set(i.cust_id.unique()), len(i)) for i in periods_2]
        periods_2 = list(itertools.chain(*periods_2))

        return *periods_2, *test_func(periods, seg_type)

