import pandas as pd
import numpy as np
import source_code.core.hypothesis_evaluation.generate_results as gen_results
import altair as alt

alt.data_transformers.disable_max_rows()

def get_fdr_power(global_stats, global_res, samples, isin, threshold):
    fdr_power = []
    
    len_ = len(global_stats)
    
    for i in range(len_):
        fdr_power.append( gen_results.get_fdr_power_results(global_stats[i] , global_res[i].copy(), samples, threshold) )
    
    fdr_csv = fdr_power[0]

    for i in range(1,len(fdr_power)):
        fdr_csv = pd.concat([fdr_csv,fdr_power[i]])
        
    nb_csv = fdr_csv[fdr_csv.Metric=='nb_resu'].groupby(['Method','Abv_method','Sample']).mean().reset_index()

    power_csv = fdr_csv[fdr_csv.Metric=='power']
    fdr_csv = fdr_csv[fdr_csv.Metric=='fdr']
    
    in_samples = [f'{int(i*100)}%' for i in samples]
    
    fdr_csv = fdr_csv[fdr_csv.Sample.isin(in_samples)]
    nb_csv = nb_csv[nb_csv.Sample.isin(in_samples)]
    power_csv = power_csv[power_csv.Sample.isin(in_samples)]
    
    fdr_csv = fdr_csv[fdr_csv.Method.isin(isin)]
    nb_csv = nb_csv[nb_csv.Method.isin(isin)]
    power_csv = power_csv[power_csv.Method.isin(isin)]

    fdr_csv['Sample'] = fdr_csv['Sample'].apply(lambda x : x.split('%')[0])
    power_csv['Sample'] = power_csv['Sample'].apply(lambda x : x.split('%')[0])
    
    power_csv.Value = power_csv.Value*100
    
    wdth = 100
    heit = 400
    
    pattern_scale = ({
       'range': ['url(#pattern_1)', 'url(#pattern_2)', 'url(#pattern_3)',
                 'url(#pattern_4)','url(#pattern_5)','url(#pattern_6)','url(#pattern_7)','url(#pattern_8)']
    })
    
    fdr_chart = alt.Chart(fdr_csv).mark_bar(stroke='black',strokeWidth=0.3,size=12).encode(
        alt.X('legend_method:N',axis=alt.Axis(title="",labels=False, titleFontSize=26, 
                                       titleFont = 'Time New Roman', titleFontStyle='italic')
             ),
        
        alt.Y('mean(Value)',
              axis=alt.Axis(title="FDR",labelFontSize=25,labelAngle=0,titleFontSize=26,grid=False,
                            titleFont = 'Time New Roman',labelFont='Time New Roman',
                            labelFontStyle='bold', titleFontStyle='italic')
             ),
        
        fill = alt.Fill('legend_method:N',title='',scale=pattern_scale,
        legend=alt.Legend(orient='bottom', titleFontSize=25, labelFontSize=23, titleFontStyle='italic',
                          labelFontStyle='bold',titleFont = 'Time New Roman',\
        labelFont='Time New Roman',symbolSize=500) )
        
    ).properties(
        width = wdth,
        height = heit,
    )

    power_chart = alt.Chart(power_csv).mark_bar(stroke='black',strokeWidth=0.3,size=12).encode(
        alt.X('legend_method:N',axis=alt.Axis(title="",labels=False, titleFontSize=16,
                                              titleFont = 'Time New Roman',titleFontStyle='italic')
             ),
        
        alt.Y('mean(Value)', scale=alt.Scale(type='pow', exponent=1/3 , domain=(0,3)),
        axis=alt.Axis(title="Power (e-2)", tickCount=4, grid=True , gridColor='white',labelFontSize=25,
                      labelAngle=0,titleFontSize=26,titleFont = 'Time New Roman',
                      labelFont='Time New Roman',labelFontStyle='bold', titleFontStyle='italic')
             ),
        
        fill = alt.Fill('legend_method:N',title='',scale=pattern_scale, 
                        legend=alt.Legend(orient='bottom', columns=4,titleFontSize=25, labelFontSize=23,
                                          labelFontStyle='bold',titleFontStyle='italic',
                                          titleFont = 'Time New Roman',labelFont='Time New Roman',symbolSize=500) 
                       )
        
    ).properties(
        width = wdth,
        height = heit
    )


    error_bars_fdr = alt.Chart(fdr_csv).mark_errorbar(ticks=True).encode(
        x='legend_method:N',
        y=alt.Y('Value',title='FDR'),
    )

    error_bars_power = alt.Chart(power_csv).mark_errorbar(ticks=True).encode(
        x='legend_method:N',
        y=alt.Y('Value',title='Power (e-2)'),
    )

    fdr_chart = alt.layer(fdr_chart, error_bars_fdr, data=fdr_csv).facet(
        column=alt.Column('Sample', title='Data Samples (%)', sort=['10','30','50','70','90','100'],
        header=alt.Header(labelFontSize=20,titleFontSize=25,titleFont = 'Time New Roman',labelFont='Time New Roman'))
    )


    power_chart = alt.layer(power_chart, error_bars_power, data=power_csv).facet(
        column=alt.Column('Sample', title='Data Samples (%)', sort=['10','30','50','70','90','100'],
        header=alt.Header(labelFontSize=20,titleFontSize=25,titleFont = 'Time New Roman',labelFont='Time New Roman'))
    )

    return (power_chart | fdr_chart).configure_legend(labelLimit= 0)

def get_cov_samples(global_stats, global_res,samples, isin, threshold):
    cov_by_samples = []

    wdth = 550
    heit = 300

    len_ = len(global_res)
    
    isin = isin+['TRAD_BY']
    
    for th in threshold:
        for i in range(len_):
            if i ==0:
                cov_samp_csv = gen_results.get_coverage_samples_results(global_stats[i], global_res[i], samples, threshold=th)
            else:
                cov_samp_csv = pd.concat( [cov_samp_csv,gen_results.get_coverage_samples_results(global_stats[i], global_res[i],\
                samples, threshold=th)] )
        
        cov_samp_csv = cov_samp_csv.groupby(['Method','Sample','legend_method'])['Coverage'].mean().reset_index()
        
        cov_samp_csv = cov_samp_csv[cov_samp_csv.Method.isin(isin)]
        
        cov_samp_csv['Sample'] = cov_samp_csv['Sample'].apply(lambda x : x.split('%')[0])
        
        if th == -1:
            th = 'Unlimited number of results (n)'
        else:
            th = f'Number of results (n) = {th}'
        
        if cov_samp_csv.Coverage.min()-0.1<0:
            mm = 0
        else:
            mm = cov_samp_csv.Coverage.min()-0.1
        
        cov_by_samples.append( alt.Chart(cov_samp_csv).mark_line(interpolate='basis',strokeWidth=2.5).encode(
            alt.X('Sample:N', axis=alt.Axis(title="Data Samples (%)",labelFontSize=25,labelAngle=20,
                                            titleFontSize=26,titleFont = 'Time New Roman',
                                            labelFont='Time New Roman',labelFontStyle='bold',
                                            titleFontStyle='italic'),
                  sort=['10','30','50','70','90','100']
                 ),
            
            alt.Y('Coverage', axis=alt.Axis(title="Coverage",labelFontSize=25,labelAngle=0,
                                            titleFontSize=26,titleFont = 'Time New Roman',
                                            labelFont='Time New Roman',labelFontStyle='bold',
                                            titleFontStyle='italic'),
                  scale=alt.Scale(domain=(mm, 1))
                 ),
            
            strokeDash=alt.StrokeDash('legend_method',title='',scale=alt.Scale( range=[ [],[25,2],[15,2],[11,2],
                                                                                       [8,2],[4,2],[2,2],
                                                                                       [1,2],[25,10],[]] ),
                                        legend=alt.Legend(orient='bottom',columns=4, titleFontSize=25, 
                                                          labelFontSize=23,labelFontStyle='bold', 
                                                          titleFontStyle='italic', titleFont = 'Time New Roman',
                                                          labelFont='Time New Roman', symbolSize=2000,
                                                          symbolStrokeWidth=5,symbolFillColor='black',
                                                          symbolStrokeColor='black')
                                       ),

            color=alt.Color('legend_method',title='', scale = alt.Scale(range=['#666666','#1F77B4','#FF7F0E','#2CA02C',
                                                                           '#D62728','#9467BD','#8C564B','#316B83',
                                                                           '#CE6DBD','#637939']),
                            legend = None
                           )
        ).properties(
            width = wdth,
            height = heit,
            title = alt.TitleParams(th, font='Time New Roman', fontSize=27,fontStyle='italic',)
        ) )

    return (cov_by_samples[0]).configure_legend(labelLimit= 0)

def get_time_sample(global_stats,samples, isin, threshold):
    time_by_samples = []

    wdth = 500
    heit = 325

    isin = isin+['TRAD_BY']
    
    len_ = len(global_stats)

    for th in threshold:
        
        for i in range(len_):
            if i == 0:
                time_samp = gen_results.get_time_samples_results(global_stats[i],samples,th)
            else:
                time_samp = pd.concat( [time_samp,gen_results.get_time_samples_results(global_stats[i],samples,th)] )

        time_samp = time_samp.groupby(['Method','Sample','legend_method'])['Time'].mean().reset_index()  
        
        time_samp = time_samp[time_samp.Method.isin(isin)]
        
        time_samp['Sample'] = time_samp['Sample'].apply(lambda x : x.split('%')[0])
        
        if th == -1:
            th = 'Unlimited number of results (n)'
        else:
            th = f'Number of results (n) = {th}'
        
        time_by_samples.append( alt.Chart(time_samp).mark_line(interpolate='basis').encode(
            
        alt.X('Sample:N', axis=alt.Axis(title="Data Samples (%)",labelFontSize=25,labelAngle=20,titleFontSize=26,
                                        titleFont = 'Time New Roman',labelFont='Time New Roman',
                                        labelFontStyle='bold', titleFontStyle='italic', titlePadding=30),
              sort=['10','30','50','70','90','100']
             ),
            
        alt.Y('Time', axis=alt.Axis(title="Response Time (sec, logscale)",labelFontSize=25,labelAngle=0,
                                    titleFontSize=26,tickCount=4, grid=True,
                                    titleFont = 'Time New Roman',labelFont='Time New Roman',
                                    labelFontStyle='bold', titleFontStyle='italic'),
             ),
            
        strokeDash=alt.StrokeDash('legend_method',title='',scale=alt.Scale( range=[ [],[25,2],[15,2],[11,2],
                                                                                       [8,2],[4,2],[2,2],
                                                                                       [1,2],[25,10],[]] ),
                                  legend=alt.Legend(orient='bottom',columns=5, titleFontSize=25,
                                                    labelFontSize=23,labelFontStyle='bold', titleFontStyle='italic',
                                                    titleFont = 'Time New Roman',labelFont='Time New Roman',
                                                    symbolSize=2000,symbolStrokeWidth=4,symbolFillColor='black',
                                                    symbolStrokeColor='black')
                                 ),
        color=alt.Color('legend_method',title='', scale = alt.Scale(range=['#666666','#1F77B4','#FF7F0E','#2CA02C',
                                                                           '#D62728','#9467BD','#8C564B','#316B83',
                                                                           '#CE6DBD','#637939']),
                           )

        ).properties(
            width = wdth,
            height = heit,
            title = alt.TitleParams(th, font='Time New Roman', fontSize=26, fontStyle='italic',)
        ) 
        )


    return alt.hconcat(time_by_samples[0]).configure_legend(labelLimit= 0)

def get_tim_n(global_stats, samples, isin):
    wdth = 500
    heit = 325

    time_by_thresh = []

    isin = isin+['TRAD_BY']
    
    len_ = len(global_stats)

    for ind,sample in enumerate(samples):
        time_n = []
        
        for i in range(len_):
            time_n.append( gen_results.get_time_n_results(global_stats[i],sample*100) )

        time_ = time_n[0]

        for i in range(1,len(time_n)):
            time_ = pd.concat([time_,time_n[i]])

        time_ = time_.groupby(['Method','n','legend_method'])['Time'].mean().reset_index()
        time_.n = time_.n.replace(['-1'], '  Unlimited')
        
        time_ = time_[time_.Method.isin(isin)]
        
        time_by_thresh.append( alt.Chart(time_).mark_line(interpolate='basis').encode(
            alt.X('n:N',axis=alt.Axis(title="Number of results (n)",labelFontSize=25,labelAngle=20,titleFontSize=26,
                                      titleFont = 'Time New Roman',labelFont='Time New Roman',
                                      labelFontStyle='bold', titleFontStyle='italic'),
                  sort=['5','10','15','20','50','100','500','Unlimited']
                 ),
            
            alt.Y('Time', axis=alt.Axis(title="Response Time (sec, logscale)",labelFontSize=25,
                                        labelAngle=0,
                                        titleFontSize=26,titleFont = 'Time New Roman',labelFont='Time New Roman',
                                        labelFontStyle='bold', titleFontStyle='italic'),
                 ),
            
            strokeDash=alt.StrokeDash('legend_method',title='',scale=alt.Scale( range=[ [],[25,2],[15,2],[11,2],
                                                                                       [8,2],[4,2],[2,2],
                                                                                       [1,2],[25,10],[]] ),
                                      legend=alt.Legend(orient='bottom',columns=5, titleFontSize=25, 
                                                        labelFontSize=23,labelFontStyle='bold', 
                                                        titleFontStyle='italic',titleFont = 'Time New Roman',
                                                        labelFont='Time New Roman',symbolSize=4000, 
                                                        symbolStrokeWidth=3.5,symbolFillColor='black',
                                                        symbolStrokeColor='black')
                                     ),
            
            color=alt.Color('legend_method',title='', scale = alt.Scale(range=['#666666','#1F77B4','#FF7F0E','#2CA02C',
                                                                           '#D62728','#9467BD','#8C564B','#316B83',
                                                                           '#CE6DBD','#637939']),
                           )

        ).properties(
            width = wdth,
            height = heit,
            title = alt.TitleParams(f'Data Samples = {samples[ind]*100}%', font='Time New Roman', fontSize=26,fontStyle='italic')
        ) )
        
    return (time_by_thresh[0]).configure_legend(labelLimit= 0)

def get_cov_p_values_n(global_stats, global_res,samples, isin, threshold):
    graphs_cov = []
    graphs_p = []
    graphs_fdr = []

    len_ = len(global_res)
    
    isin = isin+['TRAD_BY']
    
    for sample in samples:
        cov_p_values = []
        
        for i in range(len_):
            cov_p_values.append( gen_results.get_coverage_p_values_results(global_stats[i],global_res[i],sample,threshold) )

        csv_p_csv = cov_p_values[0]
        
        csv_p_csv['Step'] = 1
        csv_p_csv['Step'] = csv_p_csv.groupby(['Method'])['Step'].cumsum()

        for i in range(1,len(cov_p_values)):
            local = cov_p_values[i]
            local['Step'] = 1
            local['Step'] = local.groupby(['Method'])['Step'].cumsum()
            csv_p_csv = pd.concat([csv_p_csv,local])
        
        csv_p_csv = csv_p_csv[csv_p_csv.Method.isin(isin)]
        
        csv_p_csv = csv_p_csv.groupby(['Method','legend_method','Step'])['Coverage','p_values'].mean().reset_index()
        
        p_n_csv = csv_p_csv.groupby(['Method','Step','legend_method'])['p_values'].mean().reset_index()
        cov_n_csv = csv_p_csv.groupby(['Method','Step','legend_method'])['Coverage'].mean().reset_index()
        
        method_to_legend = cov_n_csv[['Method','legend_method']].set_index('Method')['legend_method'].to_dict()
        
        mthds = cov_n_csv.Method.unique()
        mthds_legen = [method_to_legend[i] for i in mthds]
        max_all = int(cov_n_csv.Step.max())+1

        for mtd, mthd_legen in zip(mthds,mthds_legen):
            max_mtd = int(cov_n_csv[cov_n_csv.Method==mtd].Step.max())

            loc = pd.DataFrame()
            
            loc['Step'] = np.full(max_all-max_mtd, range(max_mtd,max_all))
            loc['Coverage'] = csv_p_csv[csv_p_csv.Method==mtd].Coverage.max()
            loc['Method'] = mtd
            loc['legend_method'] = mthd_legen

            cov_n_csv = pd.concat( [ cov_n_csv,loc[['Method','Step','legend_method','Coverage']] ] )
            
            loc = pd.DataFrame()

            loc['Step'] = np.full(max_all-max_mtd, range(max_mtd,max_all))
            loc['p_values'] = p_n_csv[p_n_csv.Method==mtd]['p_values'].max()
            loc['Method'] = mtd
            loc['legend_method'] = mthd_legen
            
            p_n_csv = pd.concat( [ p_n_csv,loc[['Method','Step','legend_method','p_values']] ] )
            
        graphs_cov.append(cov_n_csv)
        graphs_p.append(p_n_csv)

    wdth = 750
    heit = 400
    
    cov_by_steps = []

    for ind, (csv_p_csv1,csv_p_csv2) in enumerate( zip(graphs_cov,graphs_p) ):

        x1 = alt.Chart(csv_p_csv1).mark_line(interpolate='basis').encode(
            alt.X('Step:Q', axis=alt.Axis(title="Number of results (n)",labelFontSize=25,labelAngle=20,titleFontSize=26,
                                          titleFont = 'Time New Roman',labelFont='Time New Roman',
                                          labelFontStyle='bold', titleFontStyle='italic')
                 ),
            
            alt.Y('Coverage', axis=alt.Axis(title="Sum Coverage",labelFontSize=25,labelAngle=0,titleFontSize=26,
                                            titleFont = 'Time New Roman',labelFont='Time New Roman',
                                            labelFontStyle='bold', titleFontStyle='italic')
                 ),

            color=alt.Color('legend_method',title='',scale = alt.Scale(range=['#666666','#1F77B4','#FF7F0E','#2CA02C',
                                                                           '#D62728','#9467BD','#8C564B','#316B83',
                                                                           '#CE6DBD','#637939']),
                            
                            legend=alt.Legend(orient='bottom',columns=2, titleFontSize=25, labelFontSize=23,
                                              labelFontStyle='bold', titleFontStyle='italic',
                                              titleFont = 'Time New Roman',labelFont='Time New Roman',
                                              symbolSize=2000,symbolStrokeWidth=5)
                           ),
        ).properties(
            width = wdth,
            height = heit,
        )
        
        x2 = alt.Chart(csv_p_csv2).mark_line(interpolate='basis', color='black').encode(
            alt.X('Step:Q', axis=alt.Axis(title="Number of results (n)",labelFontSize=25,labelAngle=20,titleFontSize=26,
                                          titleFont = 'Time New Roman',labelFont='Time New Roman',
                                          labelFontStyle='bold', titleFontStyle='italic')
                 ),
            
            alt.Y('p_values', axis=alt.Axis(title="Sum p-values",labelFontSize=25,labelAngle=0,titleFontSize=26,
                                            titleFont = 'Time New Roman',labelFont='Time New Roman',
                                            labelFontStyle='bold', titleFontStyle='italic',labelAlign='left')
                 ),
            
            strokeDash=alt.StrokeDash('legend_method',title='',scale=alt.Scale( range=[ [],[25,2],[15,2],[11,2],
                                                                                       [8,2],[4,2],[2,2],
                                                                                       [1,2],[25,10],[]] ),
                                      legend=alt.Legend(orient='bottom',columns=2, titleFontSize=25, 
                                                        labelFontSize=23,labelFontStyle='bold', 
                                                        titleFontStyle='italic',titleFont = 'Time New Roman',
                                                        labelFont='Time New Roman',symbolSize=2000,
                                                        symbolStrokeWidth=5)
                                     ),
            
        ).properties(
            width = wdth,
            height = heit,
        )
        
        cov_by_steps.append( alt.layer(x1, x2).resolve_scale( y = 'independent') )

    return cov_by_steps[0].configure_legend(labelLimit= 0)
