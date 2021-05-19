import pandas as pd
import numpy as np
import source_code.core.hypothesis_evaluation.generate_results as gen_results
import altair as alt

alt.data_transformers.disable_max_rows()

def get_fdr_power(global_stats,global_res,samples,threshold):
    fdr_power = []
    len_ = len(global_stats)

    for i in range(len_):
        fdr_power.append( gen_results.get_fdr_power_results(global_stats[i],global_res[i],samples, threshold=threshold) )

    fdr_csv = fdr_power[0]

    for i in range(1,len(fdr_power)):
        fdr_csv = pd.concat([fdr_csv,fdr_power[i]])
        
    nb_csv = fdr_csv[fdr_csv.Metric=='nb_resu'].groupby(['Method','Abv_method','Sample']).mean().reset_index()

    power_csv = fdr_csv[fdr_csv.Metric=='power']
    fdr_csv = fdr_csv[fdr_csv.Metric=='fdr']

    in_samples = [f'{i}%' for i in samples]

    fdr_csv = fdr_csv[fdr_csv.Sample.isin(in_samples)]
    nb_csv = nb_csv[nb_csv.Sample.isin(in_samples)]
    power_csv = power_csv[power_csv.Sample.isin(in_samples)]

    fdr_csv = fdr_csv[~fdr_csv.Method.isin(['COVER_G_BN','COVER_⍺_100'])]
    nb_csv = nb_csv[~nb_csv.Method.isin(['COVER_G_BN','COVER_⍺_100'])]
    power_csv = power_csv[~power_csv.Method.isin(['COVER_G_BN','COVER_⍺_100'])]

    wdth = 75
    heit = 350

    fdr_chart = alt.Chart(fdr_csv).mark_bar().encode(
        alt.X('Abv_method:N',axis=alt.Axis(title="",labelFontSize=25,labelAngle=45,titleFontSize=26,
        titleFont = 'Time New Roman',labelFont='Time New Roman',
        labelFontStyle='bold', titleFontStyle='italic')),
        
        alt.Y('mean(Value)',axis=alt.Axis(title="FDR",labelFontSize=25,labelAngle=0,titleFontSize=26,
        titleFont = 'Time New Roman',labelFont='Time New Roman',
        labelFontStyle='bold', titleFontStyle='italic')),
        
        color=alt.Color('Method:N',title='',
        scale = alt.Scale(range=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','brown']), #Blue - Orange - Green - Red - Purple
        legend=alt.Legend(orient='bottom', titleFontSize=25, labelFontSize=13,labelFontStyle='bold', titleFontStyle='italic',titleFont = 'Time New Roman',\
        labelFont='Time New Roman',symbolSize=500,symbolStrokeWidth=3),
        sort=['COVER_G_BN','COVER_G_BY','COVER_⍺_20','COVER_⍺_50','COVER_⍺_200','TRAD_BY','Perfect'])
    ).properties(
        width = wdth,
        height = heit,
    )

    power_chart = alt.Chart(power_csv).mark_bar().encode(
        alt.X('Abv_method:N',axis=alt.Axis(title="",labelFontSize=25,labelAngle=45,titleFontSize=16,
        titleFont = 'Time New Roman',labelFont='Time New Roman',
        labelFontStyle='bold', titleFontStyle='italic')),
        
        alt.Y('mean(Value)',scale=alt.Scale(type='pow',exponent=1/6,domain=(0,1.1)),
        axis=alt.Axis(title="Power",labelFontSize=25,labelAngle=0,titleFontSize=26,
        titleFont = 'Time New Roman',labelFont='Time New Roman',
        labelFontStyle='bold', titleFontStyle='italic')),
        
        color=alt.Color('Method:N',title='',
        scale = alt.Scale(range=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','brown']),#Blue - Orange - Green - Red - Purple
        legend=alt.Legend(orient='bottom', titleFontSize=25, labelFontSize=23,labelFontStyle='bold', titleFontStyle='italic',titleFont = 'Time New Roman',
        labelFont='Time New Roman',symbolSize=500,symbolStrokeWidth=3),
        sort=['COVER_G_BN','COVER_G_BY','COVER_⍺_20','COVER_⍺_50','COVER_⍺_200','TRAD_BY','Perfect']),
    ).properties(
        width = wdth,
        height = heit
    )


    error_bars_fdr = alt.Chart(fdr_csv).mark_errorbar().encode(
        x='Abv_method:N',
        y=alt.Y('Value',title='FDR'),
    )

    error_bars_power = alt.Chart(power_csv).mark_errorbar().encode(
        x='Abv_method:N',
        y=alt.Y('Value',title='Power'),
    )

    fdr_chart = alt.layer(fdr_chart, error_bars_fdr, data=fdr_csv).facet(
        column=alt.Column('Sample',sort=['10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'],
        header=alt.Header(labelFontSize=20,titleFontSize=25,titleFont = 'Time New Roman',labelFont='Time New Roman'))
    )


    power_chart = alt.layer(power_chart, error_bars_power, data=power_csv).facet(
        column=alt.Column('Sample',sort=['10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'],
        header=alt.Header(labelFontSize=20,titleFontSize=25,titleFont = 'Time New Roman',labelFont='Time New Roman'))
    )

    return power_chart | fdr_chart

def get_cov_p_values_n(global_res,samples,threshold):
    graphs_cov = []
    graphs_p = []

    len_ = len(global_res)

    for sample in samples:
        cov_p_values = []
        
        for i in range(len_):
            cov_p_values.append( gen_results.get_coverage_p_values_results(global_res[i],sample, threshold=threshold) )

        csv_p_csv = cov_p_values[0]
        
        csv_p_csv['Step'] = 1
        csv_p_csv['Step'] = csv_p_csv.groupby(['Method'])['Step'].cumsum()

        for i in range(1,len(cov_p_values)):
            local = cov_p_values[i]
            local['Step'] = 1
            local['Step'] = local.groupby(['Method'])['Step'].cumsum()
            csv_p_csv = pd.concat([csv_p_csv,local])
        
        p_n_csv = csv_p_csv.groupby(['Method','Step'])['p_values'].mean().reset_index()
        cov_n_csv = csv_p_csv.groupby(['Method','Step'])['Coverage'].mean().reset_index()

        mthds = cov_n_csv.Method.unique()
        max_all = int(cov_n_csv.Step.max())+2

        for mtd in mthds:
            max_mtd = int(cov_n_csv[cov_n_csv.Method==mtd].Step.max())

            loc = pd.DataFrame()
            
            loc['Step'] = np.full(max_all-max_mtd, range(max_mtd,max_all))
            loc['Coverage'] = csv_p_csv[csv_p_csv.Method==mtd].Coverage.max()
            loc['Method'] = mtd

            cov_n_csv = pd.concat( [ cov_n_csv,loc[['Method','Step','Coverage']] ] )
            
            loc = pd.DataFrame()

            loc['Step'] = np.full(max_all-max_mtd, range(max_mtd,max_all))
            loc['p_values'] = p_n_csv[p_n_csv.Method==mtd]['p_values'].max()
            loc['Method'] = mtd

            p_n_csv = pd.concat( [ p_n_csv,loc[['Method','Step','p_values']] ] )
            
        graphs_cov.append(cov_n_csv)
        graphs_p.append(p_n_csv)

    wdth = 750
    heit = 400

    cov_by_steps = []

    for ind, (csv_p_csv1,csv_p_csv2) in enumerate( zip(graphs_cov,graphs_p) ):

        x1 = alt.Chart(csv_p_csv1).mark_line(interpolate='basis').encode(
            alt.X('Step:Q',
            axis=alt.Axis(title="n",labelFontSize=25,labelAngle=45,titleFontSize=26,titleFont = 'Time New Roman',labelFont='Time New Roman',
            labelFontStyle='bold', titleFontStyle='italic')),
            
            alt.Y('Coverage',
            axis=alt.Axis(title="Sum Coverage",labelFontSize=25,labelAngle=0,titleFontSize=26,titleFont = 'Time New Roman',labelFont='Time New Roman',
            labelFontStyle='bold', titleFontStyle='italic')),

            color=alt.Color('Method',title='',
            scale = alt.Scale(range=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']),#Blue - Orange - Green - Red - Purple
            sort=['COVER_G_BN','COVER_G_BY','COVER_⍺_20','COVER_⍺_50','COVER_⍺_100','COVER_⍺_200'],
            legend=alt.Legend(orient='bottom',columns=2, titleFontSize=25, labelFontSize=23,labelFontStyle='bold', titleFontStyle='italic',
            titleFont = 'Time New Roman',labelFont='Time New Roman',symbolSize=2000,symbolStrokeWidth=5)),
        ).properties(
            width = wdth,
            height = heit,
        )
        
        x2 = alt.Chart(csv_p_csv2).mark_line(interpolate='basis', color='black').encode(
            alt.X('Step:Q',
            axis=alt.Axis(title="n",labelFontSize=25,labelAngle=45,titleFontSize=26,titleFont = 'Time New Roman',labelFont='Time New Roman',
            labelFontStyle='bold', titleFontStyle='italic')),
            
            alt.Y('p_values',
            axis=alt.Axis(title="Sum p-values",labelFontSize=25,labelAngle=0,titleFontSize=26,titleFont = 'Time New Roman',labelFont='Time New Roman',
            labelFontStyle='bold', titleFontStyle='italic',labelAlign='left')),
            
            strokeDash=alt.StrokeDash('Method',title='',scale=alt.Scale(range=[[25,4],[15,4],[4,4],[8,4],[]]),
            sort=['COVER_G_BN','COVER_G_BY','COVER_⍺_20','COVER_⍺_50','COVER_⍺_100','COVER_⍺_200'],
            legend=alt.Legend(orient='bottom',columns=2, titleFontSize=25, labelFontSize=23,labelFontStyle='bold', titleFontStyle='italic',
            titleFont = 'Time New Roman',labelFont='Time New Roman',symbolSize=2000,symbolStrokeWidth=5)),
        ).properties(
            width = wdth,
            height = heit,
        )
        
        cov_by_steps.append( alt.layer(x1, x2).resolve_scale( y = 'independent') )

    return cov_by_steps[0]

def get_cov_samples(global_res,samples,threshold):
    cov_by_samples = []

    wdth = 500
    heit = 300

    len_ = len(global_res)

    for th in threshold:
        cov_samples = []
        
        for i in range(len_):
            cov_samples.append( gen_results.get_coverage_samples_results(global_res[i],samples,th) )
            
        cov_samp_csv = cov_samples[0]
        cov_samp_csv = cov_samp_csv.groupby(['Method','Sample'])['Coverage'].mean().reset_index()
        
        cov_samp_csv = cov_samp_csv[~cov_samp_csv.Method.isin(['TRAD_BN','COVER_⍺_100'])]
        
        for k in range(10,110,10):
            cov_samp_csv = cov_samp_csv.replace(f'{k}.0%',f'{k}%')

        if th == -1:
            th = 'Unlimited n'
        else:
            th = f'n = {th}'
        
        if cov_samp_csv.Coverage.min()-0.1<0:
            mm = 0
        else:
            mm = cov_samp_csv.Coverage.min()-0.1
        
        cov_by_samples.append( alt.Chart(cov_samp_csv).mark_line(interpolate='basis',strokeWidth=2.5).encode(
            alt.X('Sample:N',
            axis=alt.Axis(title="Samples",labelFontSize=25,labelAngle=45,titleFontSize=26,titleFont = 'Time New Roman',labelFont='Time New Roman',
            labelFontStyle='bold', titleFontStyle='italic'),
            sort=['10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'],),
            
            alt.Y('Coverage',
                axis=alt.Axis(title="Coverage",labelFontSize=25,labelAngle=0,titleFontSize=26,titleFont = 'Time New Roman',labelFont='Time New Roman',
                labelFontStyle='bold', titleFontStyle='italic'),scale=alt.Scale(domain=(mm, 1))),
            
            strokeDash = alt.StrokeDash('Method:N',title='',scale=alt.Scale(range=[[1,4],[25,4],[15,4],[4,4],[8,4],[]]),
            sort=['COVER_G_BN','COVER_G_BY','COVER_⍺_20','COVER_⍺_50','COVER_⍺_100','COVER_⍺_200'],
            legend=alt.Legend(orient='bottom',columns=4, titleFontSize=25, labelFontSize=23,labelFontStyle='bold', titleFontStyle='italic',
            titleFont = 'Time New Roman',labelFont='Time New Roman',
            symbolSize=2000,symbolStrokeWidth=5,
            symbolFillColor='black',symbolStrokeColor='black')),

            color=alt.Color('Method',title='Method by Color',
            scale = alt.Scale(range=['#e377c2','#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']),#Pink - Blue - Orange - Green - Red - Purple
            sort=['COVER_G_BN','COVER_G_BY','COVER_⍺_20','COVER_⍺_50','COVER_⍺_100','COVER_⍺_200'],legend = None)
        ).properties(
            width = wdth,
            height = heit,
            title = alt.TitleParams(th, font='Time New Roman', fontSize=27,fontStyle='italic',)
        ) )

    return cov_by_samples[1] | cov_by_samples[0]

def get_time_sample(global_stats,samples,threshold):
    time_by_samples = []

    wdth = 500
    heit = 325

    len_ = len(global_stats)

    for th in threshold:
        time_samples = []
        
        for i in range(len_):
            time_samples.append( gen_results.get_time_samples_results(global_stats[i],samples,th) )

        time_samp = time_samples[0]

        for i in range(1,len(time_samples)):
            time_samp = pd.concat([time_samp,time_samples[i]])

        time_samp = time_samp.groupby(['Method','Sample'])['Time'].mean().reset_index()  
        
        time_samp = time_samp[~time_samp.Method.isin(['TRAD_BN','COVER_G_BN','COVER_⍺_20','COVER_⍺_100'])]

        if th == -1:
            th = 'Unlimited n'
        else :
            th = f'n = {th}'
        
        time_by_samples.append( alt.Chart(time_samp).mark_line(interpolate='basis').encode(
            
        alt.X('Sample:N',
        axis=alt.Axis(title="Samples",labelFontSize=25,labelAngle=45,titleFontSize=26,titleFont = 'Time New Roman',labelFont='Time New Roman',
        labelFontStyle='bold', titleFontStyle='italic'),sort=['10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'],),
            
        alt.Y('Time',
        axis=alt.Axis(title="Response Time",labelFontSize=25,labelAngle=0,titleFontSize=26,titleFont = 'Time New Roman',labelFont='Time New Roman',
        labelFontStyle='bold', titleFontStyle='italic') ),
            
        strokeDash=alt.StrokeDash('Method:N',title='',scale=alt.Scale(range=[[25,4],[4,4],[8,4],[]]),
        sort=['COVER_G_BN','COVER_G_BY','COVER_⍺_20','COVER_⍺_50','COVER_⍺_100','COVER_⍺_200'],
        legend=alt.Legend(orient='bottom',columns=5, titleFontSize=25, labelFontSize=23,labelFontStyle='bold', titleFontStyle='italic',
        titleFont = 'Time New Roman',labelFont='Time New Roman',symbolSize=2000,symbolStrokeWidth=4,symbolFillColor='black',symbolStrokeColor='black')),
    
        color=alt.Color('Method',title='',scale = alt.Scale(range=['#1f77b4','#2ca02c','#d62728','#9467bd']),
        sort=['COVER_G_BN','COVER_G_BY','COVER_⍺_20','COVER_⍺_50','COVER_⍺_100','COVER_⍺_200'],legend = None)

        ).properties(
            width = wdth,
            height = heit,
            title = alt.TitleParams(th, font='Time New Roman', fontSize=27,fontStyle='italic',)
        ) )


    return alt.hconcat(time_by_samples[1],time_by_samples[0])

def get_tim_n(global_stats,samples):
    wdth = 500
    heit = 325
    
    time_by_thresh = []

    len_ = len(global_stats)

    for ind,sample in enumerate(samples):
        time_n = []
        
        for i in range(len_):
            time_n.append( gen_results.get_time_n_results(global_stats[i],sample) )

        time_ = time_n[0]

        for i in range(1,len(time_n)):
            time_ = pd.concat([time_,time_n[i]])

        time_ = time_.groupby(['Method','n'])['Time'].mean().reset_index()
        time_.n = time_.n.replace(['-1'], 'Unlimited')
        
        time_ = time_[~time_.Method.isin(['TRAD_BN','COVER_G_BN','COVER_⍺_20','COVER_⍺_100'])]
        
        time_by_thresh.append( alt.Chart(time_).mark_line(interpolate='basis').encode(
            alt.X('n:N',axis=alt.Axis(title="n",labelFontSize=25,labelAngle=0,titleFontSize=26,titleFont = 'Time New Roman',labelFont='Time New Roman',
            labelFontStyle='bold', titleFontStyle='italic'),
            sort=['5','10','15','20','25','inf']),
            
            alt.Y('Time',
            axis=alt.Axis(title="Response Time",labelFontSize=25,labelAngle=0,titleFontSize=26,titleFont = 'Time New Roman',labelFont='Time New Roman',
            labelFontStyle='bold', titleFontStyle='italic') ),
            
            strokeDash=alt.StrokeDash('Method:N',title='',scale=alt.Scale(range=[[25,4],[4,4],[8,4],[]]),
            sort=['COVER_G_BN','COVER_G_BY','COVER_⍺_20','COVER_⍺_50','COVER_⍺_100','COVER_⍺_200'],
            legend=alt.Legend(orient='bottom',columns=5, titleFontSize=25, labelFontSize=23,labelFontStyle='bold', titleFontStyle='italic',
            titleFont = 'Time New Roman',labelFont='Time New Roman',symbolSize=2000,symbolStrokeWidth=4,symbolFillColor='black',symbolStrokeColor='black')),
            
            color=alt.Color('Method',title='Method by Color',scale = alt.Scale(range=['#1f77b4','#2ca02c','#d62728','#9467bd']),
            sort=['COVER_G_BN','COVER_G_BY','COVER_⍺_20','COVER_⍺_50','COVER_⍺_100','COVER_⍺_200'],legend = None)

        ).properties(
            width = wdth,
            height = heit,
            title = alt.TitleParams(f'Sample = {samples[ind]}%', font='Time New Roman', fontSize=27,fontStyle='italic')
        ) )
        
    return time_by_thresh[0] | time_by_thresh[1]