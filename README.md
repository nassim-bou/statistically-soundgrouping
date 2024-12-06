Our work is a Framework for Statistically-Sound Group Testing,
that combines coverage  and statistical testing to verify common hypotheses and return interpretable data groups.

## Prerequisites
Before you begin, ensure you have met the following requirements:
<!--- These are just example requirements. Add, duplicate or remove as required --->
* You have python3  installed  
* You have installed requirements  `pip install -r requirements.txt`
* You have jupyter notebook installed
 

## Running

- To run the data, go to ```source_code/notebooks``` and run `Run_Experiments.py`.
    * dataset : string - Name of the Dataset
    * alphas : float - Threshold of p-value
    * support : integer - Minimum number of points in a group
    * dimension : string - variable used by aggregate function
    * grouping_type : list of lists - items or users features for clustering. Each sub-list contains features of a period.
    * grouping_value : list of lists  - values of clustering features. Each sub-list contains the values of the corresponding the features in grouping_type.
    * agg_type : string - Aggregation function
    * test_args : list - [Type of test (one sample, two samples ...), Value for One Sample]
    * periods_start : list of datetime - Datetime of starting of each period
    * period_arg : list of integer - [Number of months, Number of days]
    * top_ns : list of integer- List of number of results (n)
    * period_type : string - Time_Based ('time')
    * num_hyps : list of float - Percentages of samples
    * r : string - Type of request (Ri)
    * methods : list of integer - List of method {0:TRAD, 1:COVER_G ,4:COVER_⍺, 5:β-Farsighted, 6:γ-Fixed, 7:ẟ Hopeful, 8:Ɛ-Hybrid, 9:Ψ-Support}

The execution will create results files in ```experiments\dataset\r\```.

- Use ```graphs.ipynb``` to generate the different graphs presented in the paper using the results files in ```experiments\dataset\r\```.
