# -*- coding: utf-8 -*-
"""
Created on Fri Dec 07 09:38:38 2018

"""

num_cols = ['health__pct_adult_obesity', 'health__pct_adult_smoking', 'health__pct_diabetes', 'health__pct_excessive_drinking'] 
          
def do_scatter(df, cols):
    import matplotlib.pyplot as plt
    #import statsmodels.nonparametric.smoothers_lowess as lw
    
    ## Loop over the columns and create the scatter plots
    for col in cols:
        ## first compute a lowess fit to the data
        #los = lw.lowess(df['cnt'], df[col], frac = 0.3)
    
        ## Now make the plots
        fig = plt.figure(figsize=(8, 6))
        fig.clf()
        ax = fig.gca()
        df.plot(kind = 'scatter', x = col, y = 'poverty_rate', ax = ax, alpha = 0.05)
        #plt.plot(los[:, 0], los[:, 1], axes = ax, color = 'red')
        ax.set_xlabel(col)
        ax.set_ylabel('poverty_rate')
        ax.set_title('poverty_rate vs. ' + col)    
    return 'Done'        

cat_cols = ['econ__economic_typology', 'area__urban_influence', 'area__rucc' ]
    
def do_box(df, cols):
    import matplotlib.pyplot as plt
    
    ## Loop over the columns and create the box plots
    for col in cols:
        fig = plt.figure(figsize=(8, 6))
        fig.clf()
        ax = fig.gca()
        df.boxplot(column = 'poverty_rate', by = col, ax = ax)
        ax.set_xlabel(col)
        ax.set_ylabel('poverty_rate')
        ax.set_title('poverty_rate vs. ' + col)
        plt.xticks(rotation='vertical')

        labels = [item.get_text()[0:45] for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        


    return 'Done'
    
#%matplotlib inline
do_scatter(df, num_cols)
do_box(df, cat_cols)