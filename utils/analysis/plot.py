import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def lin_reg_plot(data, meta_data, x, y, corr_result):
    """
    * data: data to do correlation (x,y에서 이름을 정해줄 것)
    * meta_data : additional data that will be appended to data, to do linear regression
    * x : (str) what column name the x should use from teh combined data/meta_data
    * y : (str) same as x, but for y
    * corr_result : correlation result ('r', 'p-val'이 있어야함)
    """
       
    #setting the data
    combined = pd.concat([data, meta_data], join = 'inner', axis = 1)
    r_corr = corr_result["r"][0]
    p_corr = corr_result["p-val"][0]
    
    #plotting
    f,ax = plt.subplots(figsize=(5,5))
    palette  = sns.color_palette("Set1", 12)

    
    sns.regplot(x=x, y= y , data=combined,scatter_kws={"s":25,"edgecolor":'k','alpha':1},line_kws={"color":'k'},color=palette[10])
    ax.set_title(f'r(%) ={r_corr.round(4)*100}, pv = {p_corr.round(9)}',fontsize=20);
    # ax.set_title(r'$\rho(\%) =-45.3$, $p \leq 0.001$',fontsize=font_size);
    plt.show()