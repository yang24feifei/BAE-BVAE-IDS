from scipy import stats
import math 
import seaborn as sns
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import random
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd
import datetime
import os
from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay
from sklearn import metrics
from pyod.models.ecod import ECOD
import copy
#import heapq

def getsortedindex(aa,bb):
    # for total uncertainty=seq(alea)+seq(epis)
    arraylen=len(aa)
    seq_a=copy.deepcopy(aa)
    seq_b=copy.deepcopy(bb)
    sortedindex_a=np.argsort(aa)  # from small to big
    sortedindex_b=np.argsort(bb)
    for i in range(0,len(sortedindex_a)):
        seq_a[sortedindex_a[i]]=i/(arraylen*2)
        seq_b[sortedindex_b[i]]=i/(arraylen*2)
    return seq_a+seq_b

#plot the metrics(FP,TP,ROC,AUC)
def rocauc(attackname,true_y, pre_y,plot=True):
    '''  y_true:真实值
    y_score：预测概率。注意：不要传入预测label！！！
    '''
    fpr,tpr,threshold=metrics.roc_curve(true_y,pre_y)
    '''
    # fpr80: the lowest fpr when tfp>0.8
    tfpind=tpr>0.8
    if len(tfpind)>0:
        fpr80=np.min(fpr[tfpind])
    else:
        fpr80=0
    #print(f'the lowest fpr when tpr>80：{fpr80}')
    '''
    roc_auc=metrics.auc(fpr,tpr) 
    try:
        roc_auc=float(roc_auc)
    except:
        print(f'float convert error:{roc_auc}')
        roc_auc=0
    if np.isnan(roc_auc):
        roc_auc=0
        
    opti_point=np.argmax(tpr-fpr)
    if plot:
        pyplot.figure(figsize=(6,6))
        pyplot.title('Validation ROC-%s'%attackname)
        pyplot.plot(fpr,tpr,'b',label='Val AUC=%0.3f'%roc_auc)
        pyplot.plot(fpr[opti_point],tpr[opti_point],marker='o',color='r',label='bes-thre=%0.3f'%threshold[opti_point])
        pyplot.legend(loc='lower right')
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("True Positive Rate")
        pyplot.xlim([0,1])
        pyplot.ylim([0,1])
        '''
        f=pyplot.gcf()
        f.savefig(attackname+'-rocauc.pdf')
        f.clear()
        ''' 
        ## only save the last outcome of each experiments.
        #attackname=attackname.split('.')[0]
        #np.save(dicoutcome4+attackname+'_roc_curve.npy',metrics.roc_curve(true_y,pre_y))

        pyplot.show()
    '''
    display = PrecisionRecallDisplay.from_predictions(true_y, pre_y, name=attackname)
    _ = display.ax_.set_title("Precision-Recall curve of %s"%attackname)
    '''
    return threshold[opti_point], fpr[opti_point] ,roc_auc,opti_point # ,threshold[np.argmax(tpr-(1-fpr))],threshold[np.argmax((1-tpr)**2+fpr**2)] # the last two value have not good performance

def threshold_search(y_true, y_proba):
    # search threshold for Accuracy
    best_threshold = 0
    best_score = 0
    #idealrate=float(1- len(np.where(y_true==1)[0])/len(y_true) )
    #beginstep=np.max((0,idealrate-0.5))
    #endstep=np.min((idealrate+0.5,1))
    for rate in np.arange(0.01,1, 0.01):
    #for rate in np.arange(beginstep, endstep, 0.01):
        threshold=np.quantile(y_proba,rate)
        y_pred=y_proba > threshold
        #score=metrics.f1_score(y_true, y_pred, average='weighted') 
        metric_report=classification_report(y_true, y_pred,output_dict=True)       
        try :
            f1_positive=metric_report['1.0']['f1-score']
        except:
            f1_positive=0
        if metric_report['accuracy'] >best_score and f1_positive!=0 :  #metric_report['macro avg']['f1-score'] 
        #if score > best_score:
            best_threshold = threshold
            best_score = metric_report['accuracy']   #metric_report['macro avg']['f1-score']
    return best_score, best_threshold

   
def plot_hist(x):
    '''
    q25, q75 = np.quantile(x, [0.25, 0.75])
    if q75>q25:
        bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
    else:
         bin_width=1 
    bins = round((np.max(x) -np.min(x)) / bin_width)
    print("Freedman–Diaconis number of bins:", bins)
    '''
    #plt.hist(x, bins=bins);82
    sns.displot(x, bins=80, kde=True,legend=False)
    
def metric_GSS(mdata):
    # actually, the lower GSS is better. we may use 1-GSS to denote its high performance
    
    mdata=np.clip(mdata, a_min=10e-8,a_max=1) # np.abs(np.max(mdata)))
    '''
    # scale data
    t = MinMaxScaler()
    t.fit(mdata)
    mdata = t.transform(mdata)
    '''
    return stats.mstats.gmean(mdata)
    #return 0
    
''' # worked in plot_metrics
def metric_GSS_gain(mdata,uncerdata,rejectiongroup):
    # Gain=W-Base ; rejectiongroup in percentage,100%
    basegss=metric_GSS(mdata)
    weightgss=[]
    weightfactor=[]
    for rej in rejectiongroup:
        threshold = np.percentile(uncerdata, rej)
        uncer_index = (uncerdata > threshold)
        
        (100-rej)* metric_GSS(mdata[uncer_index] )
    
    for rej in xx:
    # rejection probability
        threshold=np.quantile(unc_total,1-rej/100)
        pred_index=unc_total<threshold
        if len(a_prob[pred_index])>0:
            weightgss.append((1-rej/100)* metric_GSS(a_prob[pred_index]) )
            weightfactor.append((1-rej/100))
        else:
            weightgss.append(0)
            weightfactor.append(0)
        
        for i in len(weightgss):
            tempA+=weightgss[i]
            tempB+=weightfacor[i]
            
        wgss=tempA/tempB
        gaingss=wgss-basegss
'''          
def plot_metrics(modelname,typename,ascores,unc_total,y_true,rej_max=60,step=3,folder='plots',plot=True):
    ''' accuracy-rejection curve 
        rej_max: the max range of rejection rate
    '''    
    accu_list=[]
    accu2_list=[]  # from auc
    f1_list=[]
    macro_list=[]
    fpr_list=[]
    auc_list=[]
    un_list=[]
    basegss=metric_GSS(ascores)
    #print(f'base GSS is {basegss}')
    weightgss=[]
    weightauc=[]
    weightaccu=[]
    weightfpr=[]
    weightfactor=[]
    weightuncer=[]
    gss_list=[]
    wf_auc=0
    wf_accu=0
    wf_fpr=0
    wf_uncer=0   
    
    #unc_total=unc_total.to_numpy().flatten()
    size=int(np.floor(len(unc_total)*step/100))    
    sortdata=pd.DataFrame(data={'unctotal':unc_total,'ascores':ascores,'ytrue':y_true}).sort_values(by=['unctotal'],ascending=True)
    xx=np.arange(0,rej_max,step)
    unc_total2=unc_total
    ascores2=ascores
    y_true2=y_true
    
    for rej in xx:
    # rejection probability
        limit=1-rej/100
        if rej>0:
            savesize=int(np.floor(limit*len(unc_total)))
            unc_total2=sortdata.iloc[0:savesize,[True,False,False]].to_numpy().flatten()  #'unctotal']
            ascores2=sortdata.iloc[0:savesize, [False,True,False]].to_numpy().flatten() #'ascores']
            y_true2=sortdata.iloc[0:savesize,[False,False,True]].to_numpy().flatten()
        
        if len(ascores2)>0 :  #  and len(np.where(y_true2==1)[0])>0 and len(np.where(y_true2==0)[0])>0 :
            # GSS
            weightgss.append(limit* metric_GSS(ascores) )
            weightfactor.append(limit)
            gss_list.append( metric_GSS(ascores2) )
            # acurracy-rejection               
            bestthresh,bestfpr,bestauc,bestindex= rocauc(typename,y_true2, ascores2,plot=False)  
            fpr_list.append(bestfpr)
            weightfpr.append(limit*bestfpr)
            if bestfpr!=0:
                wf_fpr+=limit 
            auc_list.append(bestauc)    
            weightauc.append(limit*bestauc)
            if bestauc!=0:
                wf_auc+=limit
            un_list.append(unc_total2[bestindex])
            weightuncer.append(limit*unc_total2[bestindex])
            if unc_total2[bestindex]!=0:
                wf_uncer+=limit
            anomalies=ascores2 > bestthresh
            metric_report=classification_report(y_true2, anomalies,output_dict=True)      # len(metric_report)=5  
            accu2_list.append(metric_report['accuracy'] )
            '''    
            bestaccu,bestthresh= threshold_search(y_true2,ascores2)
            anomalies=ascores2 > bestthresh
            metric_report=classification_report(y_true2, anomalies,output_dict=True)      # len(metric_report)=5        
            '''
            accu_list.append(metric_report['accuracy'] )
            weightaccu.append(limit*metric_report['accuracy'])
            if metric_report['accuracy']!=0:
                wf_accu+=limit
            macro_list.append(metric_report['macro avg']['f1-score'])
            try:
                f1_list.append(metric_report['1.0']['f1-score'])
            except :
                f1_list.append(0)
            
        else:
            weightgss.append(0)
            weightfactor.append(0)
            weightfpr.append(0)
            weightauc.append(0)
            weightaccu.append(0)
            gss_list.append(0)
            fpr_list.append(0)
            auc_list.append(0)   
            accu_list.append(0)            
            accu2_list.append(0)
            macro_list.append(0)
            f1_list.append(0)
            un_list.append(0)
            
    auclist=np.asarray(auc_list)
    unlist=np.asarray(un_list)
    gsslist=np.asarray(gss_list)
    
    wf,wfpr,wauc,waccu=0,0,0,0  # ,wgss,gaingss,0,0
    wf=np.sum(weightfactor)
    wfpr= np.sum(weightfpr)/wf_fpr 
    wauc= np.sum(weightauc)/wf_auc   
    waccu= np.sum(weightaccu)/wf_accu
    wuncer=np.sum(weightuncer)/wf_uncer
    wgss=np.sum(weightgss)/wf      # wg/wf   #measures the weighted average accuracy
    gaingss=wgss-basegss   # measures the average improvement in accuracy
    gainaccu=waccu-accu_list[0]
    gainauc=wauc-auc_list[0]
    avgaccu2=  np.mean(accu2_list)
    avgauc=np.mean(auc_list)
    gainavgaccu2=avgaccu2-accu2_list[0]
    gainavgauc=avgauc-auc_list[0]
    return_dict= { "wfpr":wfpr, 
                      "wauc":wauc,
                      "waccu":waccu,
                  "avgaccu2":avgaccu2 ,
                  "avgauc":avgauc,
                  "wuncer":wuncer,
                     # "wgss":wgss,
               # "gaingss":gaingss,
                  "gainavgaccu":gainavgaccu2,
                  "gainavgauc":gainavgauc,
                  "gainaccu":gainaccu,
                  "gainauc":gainauc,
                  "minfpr":np.min(fpr_list),
                  "maxfpr":np.max(fpr_list),
                  "maxauc":np.max(auc_list),
                  "minauc":np.min(auc_list),
                  "maxaccu":np.max(accu_list),
                  "minaccu":np.min(accu_list),
                  "minuncer":np.min(un_list),
                  "maxuncer":np.max(un_list),
                  #"bestgss":np.max(gss_list)
                }
    
    nowtime=datetime.datetime.now()
    nowtime=str(nowtime.year)+str(nowtime.month)+str(nowtime.day)+str(nowtime.hour)
    if plot:            
        fig, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = pyplot.subplots(3, 2,figsize=(8,8),sharex=True,sharey=True)  #,constrained_layout=True)
        fig.suptitle(f'performance following rejection rate of {typename} {modelname}')
        ax1.plot(xx, auc_list)
        ax1.set_title('AUC')
        ax1.fill_between(xx, auclist-unlist, auclist+unlist ,alpha=0.3, facecolor='grey')
        #ax1.errorbar(xx, auc_list, yerr=un_list, fmt='.',color='grey')
        ax1.set(ylabel='Percent')
        ax2.plot(xx,gss_list)
        ax2.set_title('GSS')
        ax2.fill_between(xx,gsslist-unlist,gsslist+unlist,alpha=0.3,facecolor='grey')

        ax3.plot(xx, fpr_list, 'tab:orange')
        ax3.set_title('FPR')
        ax4.plot(xx, accu_list, 'tab:green')
        ax4.set_title('Accuracy')
        #ax4.set(xlabel='Rejection', ylabel='Percent')

        ax5.plot(xx, macro_list, 'tab:red')
        ax5.set_title('macro F1') 
        ax5.set(xlabel='Rejection', ylabel='Percent')
        ax6.plot(xx,f1_list, 'tab:purple')
        ax6.set_title('F1')
        ax6.set(xlabel='Rejection')

        for ax in fig.get_axes():
            ax.label_outer()
            ax.set_xlim(0,rej_max)
            ax.set_ylim(0,1)        
    
        if os.path.exists(folder) == False:
            os.mkdir(folder)    
        pyplot.savefig(folder+'/'+modelname+'+'+typename+'+'+nowtime+'+'+"allmetrics.png")

    save_path = folder+"/"+modelname+'+'+'allmetrics'+".csv"
    csv_exists = os.path.exists(save_path)
    csv_mode = 'a' if csv_exists else 'w'
    #header_mode = False if csv_exists else True
    if csv_mode=='w':
        save_data = pd.DataFrame.from_dict([xx])
        save_data.to_csv(save_path, mode=csv_mode, header=True , index_label=nowtime,index=True)
        csv_mode='a'
    save_data = pd.DataFrame.from_dict([xx])
    save_data.to_csv(save_path, mode=csv_mode, header=True, index_label=typename+'+'+nowtime,index=True)
    save_data = pd.DataFrame.from_dict([auc_list])
    save_data.to_csv(save_path, mode=csv_mode, header=xx, index_label='AUC',index=True)
    save_data = pd.DataFrame.from_dict([accu2_list])
    save_data.to_csv(save_path, mode=csv_mode, header=xx, index_label='Accu2',index=True)
    save_data = pd.DataFrame.from_dict([fpr_list])
    save_data.to_csv(save_path, mode=csv_mode, header=xx, index_label='FPR',index=True)
    save_data = pd.DataFrame.from_dict([accu_list])
    save_data.to_csv(save_path, mode=csv_mode, header=xx, index_label='ACCU',index=True)
    save_data = pd.DataFrame.from_dict([macro_list])
    save_data.to_csv(save_path, mode=csv_mode, header=xx, index_label='MacroF1',index=True)    
    save_data = pd.DataFrame.from_dict([f1_list])
    save_data.to_csv(save_path, mode=csv_mode, header=xx, index_label='F1',index=True)
    save_data = pd.DataFrame.from_dict([un_list])
    save_data.to_csv(save_path, mode=csv_mode, header=xx, index_label='Uncer',index=True)
    
    save_data = pd.DataFrame.from_dict([return_dict])
    save_data.to_csv(save_path, mode=csv_mode, header=True, index_label='best',index=True)
    
    del accu_list,auc_list, f1_list, macro_list,fpr_list, un_list,  weightauc,weightaccu,    weightfpr,  weightfactor, gss_list, auclist,gsslist,unlist,ascores,unc_total,y_true, wf_auc,wf_accu, wf_fpr,wf_uncer  # ,basegss, weightgss,
    
    return return_dict

def UQ_ale_epi_Kwon(mdata):
    # Kwon(2020) 
    if mdata.ndim>1:
        # eq.7,8 in https://openreview.net/pdf?id=Sk_P2Q9sG
        # see https://github.com/ykwon0407/UQ_BNN/issues/1
        alea = np.mean(mdata*(1-np.absolute(mdata)), axis=1)
        epi = np.mean(mdata**2, axis=1) - np.mean(mdata, axis=1)**2 
    else:
        alea = mdata*(1-np.absolute(mdata))   
        epi =mdata**2 - np.mean(mdata)**2     
        
    return alea,epi

def UQ_ale_epi_Kendall(mu,std):
    # Kendall and Gal(2017) 
    # estimate uncertainties (eq. 4 )
    # eq.4 in https://openreview.net/pdf?id=Sk_P2Q9sG
    
    # scale data
    t = MinMaxScaler()
    t.fit(mu)
    mu = t.transform(mu)
    t.fit(std)
    std=t.transform(std)
    
    if mu.ndim>1:
        alea = np.mean(std**2, axis=1)
        epi = np.mean(mu**2, axis=1) - np.mean(mu, axis=1)**2
    else:
        alea =std**2
        epi = mu**2- np.mean(mu)**2
    return alea,epi
    
# paper [27] , transfer anomaly score to anomaly probability
#****************************************************************1.scalling**********************************
def regular_linear(ascores):
    ''' Linear Inversion 
    "can be interpreted as assuming a uniform distribution"  '''
    smax=np.max(ascores)
    return smax-ascores

def regular_log(ascores):
    '''Logarithmic inversion
    ascores: anomaly scores, which must be greater than zero, and finite.   '''
    smax=np.max(ascores)
    return -np.log(ascores/smax)
#***************************************************************2.add distribution******************************
def add_gaussian(ascore):
    '''when ascore is 1-D size. '''
    mean=np.mean(ascore)
    var=np.var(ascore)
    items=[]
    for i in range(len(ascore)):
        items.append( random.gauss(mean,var) )
    items=np.asarray(items)
    return items

def add_gamma(ascore):
    '''when ascore is 1-D size. '''
    mean=np.mean(ascore)
    var=np.var(ascore)
    """Yields a list of random numbers following a gamma distribution defined by mean and variance"""
    g_alpha = mean*mean/var
    g_beta = mean/var
    asdist=[]
    #stats.gamma.rvs(g_alpha,size=mean.size) 
    for i in range(len(ascore)):
        asdist.append( random.gammavariate(g_alpha,1/g_beta)   )   
    asdist=np.asarray(asdist)
    return asdist

def add_triang(ascore):
    '''when ascore is 1-D size. '''
    mean=np.mean(ascore)  
    items=stats.triang.rvs(mean,size=ascore.size) 
    items=np.asarray(items)
    return items

def add_poisson(ascore):
    '''when ascore is 1-D size. '''
    mean=np.mean(ascore)  
    items=stats.poisson.rvs(mean,size=ascore.size)     
    items=np.asarray(items)
    return items

def add_t(ascore):
    '''when ascore is 1-D size. '''
    mean=np.mean(ascore)  
    items=stats.t.rvs(df=mean,size=ascore.size) 
    items=np.asarray(items)
    return items

def add_uniform(ascore):
    '''when ascore is 1-D size. '''
    mean=np.mean(ascore)  
    items=stats.uniform.rvs(mean,size=ascore.size) 
    items=np.asarray(items)
    return items

## ********************************************************3. Normalization***************************************
# cdf() suit to known distribution
def normalization_gaussian_1(ascores):
    ''' using cdf(mu)=0.5, cdf(max)=1 ,range of [0,1]'''
    norm=stats.norm.cdf(ascores)*2-1
    #norm[np.where(norm<0)]=0
    norm=np.absolute(norm)
    return norm   

def normalization_gaussian_2(ascores):
    ''' using cdf(),  origin range of [-1,1]; return range [0,1] after MinMaxScaler()'''
    mu_score=np.mean(ascores)
    sigma_score=np.var(ascores)
    norm=stats.norm.cdf(ascores,mu_score,sigma_score)-stats.norm.cdf(mu_score,mu_score,sigma_score)/ (1-stats.norm.cdf(mu_score,mu_score,sigma_score))
    norm=np.absolute(norm)
    '''
    t = MinMaxScaler()
    t.fit(norm.reshape(-1,1))
    norm = t.transform(norm.reshape(-1,1))
    '''
    return norm   # norm.reshape(1,-1)[0]

def normalization_gaussian_3(ascores):
    ''' using erf(), origin range of [-1,1]; return range [0,1] after MinMaxScaler()'''
    mu_score=np.mean(ascores)
    sigma_score=np.var(ascores)
    for i in range(0,len(ascores)):
        norm=math.erf((ascores[i]-mu_score)/ (sigma_score*np.sqrt(2) ))
        ascores[i]=norm
    
    t = MinMaxScaler()
    t.fit(ascores.reshape(-1,1))
    ascores = t.transform(ascores.reshape(-1,1))
    return ascores.reshape(1,-1)[0] 

def normalization_gaussian_custom(ascores, basescore):
    ''' condition on basescore,  origin range of [-1,1]; return range [0,1] after MinMaxScaler()'''
    mu_score=basescore
    sigma_score=np.sqrt(np.mean(np.square(ascores-basescore)))
    for i in range(0,len(ascores)):
        norm=math.erf((ascores[i]-mu_score)/ (sigma_score*np.sqrt(2) ))
        if norm<0:
            norm=np.absolute(norm)
        ascores[i]=norm
    '''
    t = MinMaxScaler()
    t.fit(ascores.reshape(-1,1))
    ascores = t.transform(ascores.reshape(-1,1))
    '''
    return ascores  # ascores.reshape(1,-1)[0]
    
def normalization_gamma(ascores):
    ''' range of origin range of [-1,1]; return range [0,1] after MinMaxScaler()'''
    mu_score=np.mean(ascores)
    sigma_score=np.var(ascores)
    kk=mu_score**2/ sigma_score**2
    theta=sigma_score/ mu_score**2
    #norm= (stats.gamma.cdf(ascores,a=kk, loc=ascores/theta, scale=3)-stats.gamma.cdf(mu_score,a=kk, loc=ascores/theta,scale=3) )/ ( 1-stats.gamma.cdf(mu_score,a=kk, loc=ascores/theta,scale=3))   # scale=1
    norm= (stats.gamma.cdf(ascores,a=kk, loc=0, scale=1)-stats.gamma.cdf(mu_score,a=kk, loc=0,scale=1) )/ ( 1-stats.gamma.cdf(mu_score,a=kk, loc=0,scale=1))   # scale=1
    t = MinMaxScaler()
    t.fit(norm.reshape(-1,1))
    ascores = t.transform(norm.reshape(-1,1))
    return norm.reshape(1,-1)[0]

def normalization_triang(ascores):
    #''' range of origin range of [-1,1]; return range [0,1] after MinMaxScaler()'''
    mu_score=np.mean(ascores)
    #sigma_score=np.var(ascores)
    norm= (stats.triang.cdf(ascores,c=mu_score, loc=0, scale=1)-stats.triang.cdf(mu_score,c=mu_score, loc=0,scale=1) )/ ( 1-stats.triang.cdf(mu_score,c=mu_score, loc=0,scale=1))       
    norm=np.nan_to_num(norm)
    norm=np.absolute(norm)
    
    t = MinMaxScaler()
    t.fit(norm.reshape(-1,1))
    norm = t.transform(norm.reshape(-1,1))
    norm=np.squeeze(norm)
    
    return norm 

def normalization_poisson(ascores):
    #''' range of origin range of [-1,1]; return range [0,1] after MinMaxScaler()'''
    mu_score=np.mean(ascores)
    #sigma_score=np.var(ascores)
    norm= (stats.poisson.cdf(ascores,mu=mu_score)-stats.poisson.cdf(mu_score,mu=mu_score) )/ ( 1-stats.poisson.cdf(mu_score,mu=mu_score))       
    norm=np.absolute(norm)
    
    t = MinMaxScaler()
    t.fit(norm.reshape(-1,1))
    norm = t.transform(norm.reshape(-1,1))    
    norm=np.squeeze(norm)
    return norm 

def normalization_t(ascores):
    #''' range of origin range of [-1,1]; return range [0,1] after MinMaxScaler()'''
    mu_score=np.mean(ascores)
    #sigma_score=np.var(ascores)
    norm= (stats.t.cdf(ascores,df=mu_score, loc=0, scale=1)-stats.t.cdf(mu_score,df=mu_score, loc=0,scale=1) )/ ( 1-stats.t.cdf(mu_score,df=mu_score, loc=0,scale=1))    
    norm=np.nan_to_num(norm)   
    norm=np.absolute(norm)
    
    t = MinMaxScaler()
    t.fit(norm.reshape(-1,1))
    norm = t.transform(norm.reshape(-1,1))
    norm=np.squeeze(norm)
    
    return norm 

def normalization_uniform(ascores):
    #''' range of origin range of [-1,1]; return range [0,1] after MinMaxScaler()'''
    mu_score=np.mean(ascores)
    #sigma_score=np.var(ascores)
    norm= (stats.uniform.cdf(ascores,loc=0, scale=1)-stats.uniform.cdf(mu_score, loc=0,scale=1) )/ ( 1-stats.uniform.cdf(mu_score, loc=0,scale=1))         
    norm=np.nan_to_num(norm)
    norm=np.absolute(norm)
    
    t = MinMaxScaler()
    t.fit(norm.reshape(-1,1))
    norm = t.transform(norm.reshape(-1,1))    
    norm=np.squeeze(norm)
    return norm 


def normalization_ECDF(ascores):
    '''pure ecdf, no normalization. suit to unknown distribution, e.g. multi peaks distribution'''
    norm=ECDF(ascores)
    return norm (ascores)  

def normalization_ECDF_2(ascores):
    ''' normalization using ecdf(),  '''
    mu_score=np.mean(ascores)
    norm=ECDF(ascores)
    norm=norm(ascores)-norm(mu_score)/ (1-norm(mu_score))
    norm=np.absolute(norm)
    return norm

def normalization_ECOD(ascores):
    ''' normalization using ecod() from github,  '''
    singledim=False
    if ascores.ndim==1:
        ascores=ascores.reshape(-1,1)
        singledim=True
    # train an ECOD detector
    clf = ECOD()
    clf.fit(ascores)
    # get outlier scores
    ecodscore=clf.decision_scores_  # raw outlier scores on the train data
    #if singledim==True:
    #    ecodscore=ecodscore.reshape(1,-1)
    return ecodscore

    #y_test_scores = clf.decision_function(X_test)  # predict raw outlier scores on test
    '''
    @article{zhao2019pyod,
      author  = {Zhao, Yue and Nasrullah, Zain and Li, Zheng},
      title   = {PyOD: A Python Toolbox for Scalable Outlier Detection},
      journal = {Journal of Machine Learning Research},
      year    = {2019},
      volume  = {20},
      number  = {96},
      pages   = {1-7},
      url     = {http://jmlr.org/papers/v20/19-011.html}
    }
    '''
#**********************************************************************************************************************************


def plot_UQ_density(alea,epis,thres):
    ''' plot density of larger alea than thres '''
    print('aleatoric mean: ', np.mean(alea))
    print('epistemic mean: ', np.mean(epis))
    
    threshold = np.percentile(alea, thres)
    alea_index = (alea > threshold)#
    if len(alea_index[alea_index])>0:
        if np.min(epis[alea_index])==np.max(epis[alea_index]) or np.min(alea[alea_index])==np.max(alea[alea_index]) or np.min(epis[alea_index]+alea[alea_index]) == np.max(epis[alea_index]+alea[alea_index]):
            print(f'There is only one same value in epis[alea_index].')
        else:
            pyplot.figure(figsize=(6,6))
            try:
                #data = np.vstack([list_alea[alea_index], list_epis[alea_index]]).T   data, 
                ax = sns.kdeplot(x=alea[alea_index],y=epis[alea_index], shade = True, cmap = "gray", cbar=False,common_norm=False)
                ax.patch.set_facecolor('white')
                ax.collections[0].set_alpha(0)
                ax.set_xlabel('Aleatoric', fontsize = 10)
                ax.set_ylabel('Epistemic', fontsize = 10)
                #ax.set_xlim(np.min(alea[alea_index])-0.05, np.max(alea[alea_index])+0.05)  
                #ax.set_ylim(np.min(epis[alea_index])-0.05, np.max(epis[alea_index])+0.05)  
                #pyplot.savefig('./fig/epis_vs_alea.pdf')
            except Exception as e:
                print(e)
                pyplot.hist(alea[alea_index])
                pyplot.hist(epis[alea_index])
                    
            pyplot.show()
    else:
        print(f'size of alearotic equals 0 when threshold={thres}')
        
def predict_UQ(predmodel,testdata,T, uqname='Kwon'):
    ''' T : repeat times '''
    #alea_list = []
    #epis_list = []
    mu_recon=np.zeros(testdata.shape)
    std_recon=np.zeros(testdata.shape)
    y_recon=np.zeros(testdata.shape)
       
    for i in range(0,T):
        outp=predmodel.predict(testdata)
        if len(outp)==3:  # for Tensorflow models
            test_mu,test_std,test_recon=outp[0],outp[1],outp[2]
        elif len(outp)==1 or len(outp)>20:
            test_recon=outp
            test_mu=np.mean(test_recon)
            test_std=np.var(test_recon)
        else:  # for BangXY models
            test_recon=outp['y_pred']  # 'se','bce'
            test_mu=np.mean(test_recon)
            test_std=np.var(test_recon)            
            
        mu_recon+=test_mu
        std_recon+=test_std
        y_recon+=test_recon
    
    test_mu= mu_recon/T
    test_std= std_recon/T
    test_recon=y_recon/T
    
    if uqname=='Kwon':
        aleatoric , epistemic = UQ_ale_epi_Kwon(test_recon)
    elif uqname=='Kendall':
        aleatoric , epistemic = UQ_ale_epi_Kendall(test_mu,test_std)
        
    return test_mu,test_std,test_recon, aleatoric, epistemic
'''
    if uqname=='Kwon':
        aleatoric , epistemic = UQ_ale_epi_Kwon(test_recon)
    elif uqname=='Kendall':
        aleatoric , epistemic = UQ_ale_epi_Kendall(test_mu,test_std)

    alea_list.append(aleatoric)
    epis_list.append(epistemic)

print('aleatoric: ', np.mean(alea_list), np.std(alea_list))
print('epistemic: ', np.mean(epis_list), np.std(epis_list)) 

alea_single=alea_list[0]
for i in range(1,len(alea_list)):
    alea_single+=alea_list[i]
alea_single=alea_single/len(alea_list)

epis_single=epis_list[0]
for i in range(1,len(epis_list)):
    epis_single+=epis_list[i]
epis_single=epis_single/len(epis_list)

# why uncertainty <0 ?
alea_single[np.where(alea_single<0)]=1
epis_single[np.where(epis_single<0)]=1

return mu_recon/T, std_recon/T,y_recon/T, alea_single, epis_single
'''
def get_error_term(v1,v2, _rmse=True):
    if _rmse:
        return np.sqrt(np.mean((v1-v2)**2,axis=1))
    else:        #return MAE
        return np.mean(abs(v1-v2),axis=1)
    

            
        
            
            