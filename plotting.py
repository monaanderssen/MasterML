import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pylatex
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, classification_report, make_scorer
from samples import bkg_dic
"""
import matplotlib
matplotlib.use('PS')
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['text.latex.unicode']=True
"""


"""

def plotTrainTestOutput(X_train):
  backgrounds = [363355, 341421, 364280, 410644, 304014, 364242, 410470, 364156, 364100]

  for i in backgrounds:
    plt.hist(X_train.query('ylabel==0 & DatasetNumber==' + str(i)).loc[:, 'output'], 20, stacked=True, weights = X_train.query('ylabel==0 & DatasetNumber==' + str(i)).loc[:,'eventweight'])


  #ax.set(xlabel=xlabelText, ylabel="Entries", yscale="log", title = figuretext)
  #sns.set(font= 'stixgeneral')

  #handles, labels = plt.gca().get_legend_handles_labels()
  #leg_train = mpatches.Patch(color='black', linestyle='solid', fill=True, alpha=0.5, label='train')
  #leg_test = mpatches.Patch(color='black', linestyle='dashed', fill=False, alpha=0.5, label='test')
  #plt.legend(handles=[leg_train, leg_test, handles[0], handles[1]], labels=["train", "test", labels[0], labels[1]], ncol=2)

  return


#def plotFinalTestoutput(X_test_bkg, ew_test_bkg, X_test_sig, ew_test_sig, figure_text=''):
#  plt.hist()


def plotFinalTestOutput(X_test_bkg, ew_test_bkg, X_test_sig, ew_test_sig, figure_text, xlabelText):
  ax = sns.distplot(X_test_bkg, #X_test.query("ylabel==0").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_test_bkg #X_test.query("ylabel==0").loc[:,"eventweight"]
                              }, 
                    color="tab:blue",
                    label="bkg")
  sns.distplot(X_test_sig, #X_test.query("ylabel==1").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_test_sig #X_test.query("ylabel==1").loc[:,"eventweight"]
                              }, 
                    color="tab:orange",
                    label="sig", 
                    ax=ax)
  ax.set(xlabel=xlabelText, ylabel="Entries", yscale="log", title = figure_text)
  sns.set(font= 'stixgeneral')
  #if figure_text:
  #  plt.rc('text', usetex='True')
  #  plt.text(0.5, 0.9, figure_text)
  plt.legend()

  return


def plotFinalTestOutputWithData(X_test_bkg, ew_test_bkg, X_test_sig, ew_test_sig, output_data,ew_data, figure_text, xlabelText):
  ax = sns.distplot(X_test_bkg, #X_test.query("ylabel==0").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_test_bkg #X_test.query("ylabel==0").loc[:,"eventweight"]
                              }, 
                    color="tab:blue",
                    label="bkg")
  
  sns.distplot(X_test_sig, #X_test.query("ylabel==1").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_test_sig #X_test.query("ylabel==1").loc[:,"eventweight"]
                              }, 
                    color="tab:orange",
                    label="sig")
  sns.distplot(output_data, #X_test.query("ylabel==1").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_data #X_test.query("ylabel==1").loc[:,"eventweight"]
                              }, 
                    color="tab:green",
                    label="data", 
                    ax=ax)
  ax.set(xlabel=xlabelText, ylabel="Entries", yscale="log", title = figure_text)
  sns.set(font= 'stixgeneral')
  
  #if figure_text:
  #  plt.rc('text', usetex='True')
  #  plt.text(0.5, 0.9, figure_text)
  plt.legend()

  return


def plotSetup(results_path, X_train, X_test, model):

  plt.figure(1)
  plotTrainTestOutput(X_train)
  plt.savefig(results_path + 'rawPlot.pdf')
  # Plot weighted train and test output, with test set multiplied by 2 to match number of events in training set

            
  # Plot feature importance
  print("model.feature_importances_", model.feature_importances_)
  print("np.sum(model.feature_importances_)", np.sum(model.feature_importances_))
  s_feat_importance = pd.Series(model.feature_importances_, index=X_train.drop(["DatasetNumber", "eventweight", "ylabel", "output"], axis=1).columns)
  print("X_train.drop(['eventweight', 'ylabel'], axis=1).columns\n", X_train.drop(["eventweight", "DatasetNumber", "ylabel", "output"], axis=1).columns)
  
  s_feat_importance.rename({"nBJet20_MV2c10_FixedCutBEff_77" : "b-jets"}, inplace=True)
    
  plt.figure(4)
  sns.set(style="ticks", color_codes=True)
  ax = sns.barplot(x=s_feat_importance*100, y=s_feat_importance.index, ci = None)#, palette="Blues_r")
  #ax.set_yticklabels(s_feat_importance.index)
  ax.set(xlabel="Feature importance [%]")
  sns.set(font= 'stixgeneral')
  plt.savefig(results_path + 'featureImportance.pdf')
  
  # Plot ROC curve
  plt.figure(5)
  fpr, tpr, thresholds = metrics.roc_curve(X_test.loc[:,"ylabel"], X_test.loc[:,"output"])
  auc = metrics.roc_auc_score(X_test.loc[:,"ylabel"], X_test.loc[:,"output"])
  ax = sns.lineplot(x=fpr, y=tpr, estimator=None, label='ROC curve: AUC = %0.2f' % auc)
  plt.plot([0,1], [0,1], linestyle="--")
  ax.set(xlabel="False positive rate", ylabel="True positive rate")
  sns.set(font= 'stixgeneral')
  plt.savefig(results_path + 'ROCcurve.pdf')

  #plt.show()
  return

def plotLoop(results_path, datasetlist, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext):
  plt.figure(datasetlist[i])
  plotTrainTestOutput(X_train)
  plt.savefig(results_path + 'scaled_train_test_'+ datasetlist[i]+'.pdf')
  # Plot final signal vs background estimate for test set, scaled to 10.6/fb
  plt.clf()
  plt.figure(datasetlist[i])
  plotFinalTestOutput(X_test.query("ylabel==0").loc[:,"output"],
                      3*X_test.query("ylabel==0").loc[:,"eventweight"],
                      X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"output"],
                      3*X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"eventweight"],
                      figuretext, 'XGBoost output')#,
  #figure_text='10~fb$^{-1}$')
  plt.savefig(results_path + 'test_'+ datasetlist[i]+'.pdf')
  plt.cla()

  return



def plotSetupNN(results_path, X_train, X_test, model):

  plt.figure(1)
  plotTrainTestOutput(X_train.query("ylabel==0").loc[:,"output"], None,
                      X_train.query("ylabel==1").loc[:,"output"], None,
                      X_test.query("ylabel==0").loc[:,"output"], None,
                      X_test.query("ylabel==1").loc[:,"output"], None,
                      'Raw output from training', 'Neural network output')
  plt.savefig(results_path + 'rawPlot.pdf')
  # Plot weighted train and test output, with test set multiplied by 2 to match number of events in training set
  
  # Plot ROC curve
  plt.figure(5)
  fpr, tpr, thresholds = metrics.roc_curve(X_test.loc[:,"ylabel"], X_test.loc[:,"output"])
  auc = metrics.roc_auc_score(X_test.loc[:,"ylabel"], X_test.loc[:,"output"])
  ax = sns.lineplot(x=fpr, y=tpr, estimator=None, label='ROC curve: AUC = %0.2f' % auc)
  plt.plot([0,1], [0,1], linestyle="--")
  ax.set(xlabel="False positive rate", ylabel="True positive rate")
  plt.savefig(results_path + 'ROCcurve.pdf')

  #plt.show()
  return

def plotLoopNN(results_path, datasetlist, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext):
  plt.figure(datasetlist[i])
  plotTrainTestOutput(X_train.query("ylabel==0").loc[:,"output"],
                      #sum_bkg_scaled_weights_train*X_train.query("ylabel==0").loc[:,"eventweight"],
                      scale_fac*X_train.query("ylabel==0").loc[:,"eventweight"],
                      X_train.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"output"],
                      X_train.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"eventweight"],
                      X_test.query("ylabel==0").loc[:,"output"],
                      #sum_bkg_scaled_weights_train*sum_bkg_scaled_weights*X_test.query("ylabel==0").loc[:,"eventweight"],
                      2.34*X_test.query("ylabel==0").loc[:,"eventweight"],
                      X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"output"],
                      #sum_sig_scaled_weights*X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"eventweight"],
                      2.34*X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"eventweight"],
                      figuretext, 'Neural network output')
  plt.savefig(results_path + 'scaled_train_test_'+ datasetlist[i]+'.pdf')
  # Plot final signal vs background estimate for test set, scaled to 10.6/fb
  plt.clf()
  plt.figure(datasetlist[i])
  plotFinalTestOutput(X_test.query("ylabel==0").loc[:,"output"],
                      3.34*X_test.query("ylabel==0").loc[:,"eventweight"],
                      X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"output"],
                      3.34*X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"eventweight"],
                      figuretext, 'Neural network output')#,
  #figure_text='10~fb$^{-1}$')
  plt.savefig(results_path + 'test_'+ datasetlist[i]+'.pdf')
  plt.cla()

  return




















"""
def plotTrainTestOutput(X_train_bkg, ew_train_bkg, X_train_sig, ew_train_sig, X_test_bkg, ew_test_bkg, X_test_sig, ew_test_sig, figuretext, xlabelText):
  ax = sns.distplot(X_train_bkg, #X_train.query("ylabel==0").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_train_bkg #X_train.query("ylabel==0").loc[:,"eventweight"]
                              }, 
                    color="tab:blue",
                    label="bkg")
  sns.distplot(X_train_sig, #X_train.query("ylabel==1").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_train_sig #X_train.query("ylabel==1").loc[:,"eventweight"]
                              }, 
                    color="tab:orange",
                    label="sig", 
                    ax=ax)
  sns.distplot(X_test_bkg, #X_test.query("ylabel==0").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"step", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"dashed", 
                              "linewidth":2, 
                              'alpha':1, 
                              "weights":ew_test_bkg #X_test.query("ylabel==0").loc[:,"eventweight"]
                              }, 
                    color="tab:blue",
                    #label="bkg test", 
                    ax=ax)
  sns.distplot(X_test_sig, #X_test.query("ylabel==1").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"step", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"dashed", 
                              "linewidth":2, 
                              'alpha':1, 
                              "weights":ew_test_sig #X_test.query("ylabel==1").loc[:,"eventweight"]
                              }, 
                    color="tab:orange",
                    #label="sig test", 
                    ax=ax)
  ax.set(xlabel=xlabelText, ylabel="Entries", yscale="log", title = figuretext)
  sns.set(font= 'stixgeneral')

  handles, labels = plt.gca().get_legend_handles_labels()
  leg_train = mpatches.Patch(color='black', linestyle='solid', fill=True, alpha=0.5, label='train')
  leg_test = mpatches.Patch(color='black', linestyle='dashed', fill=False, alpha=0.5, label='test')
  plt.legend(handles=[leg_train, leg_test, handles[0], handles[1]], labels=["train", "test", labels[0], labels[1]], ncol=2)

  return


#def plotFinalTestoutput(X_test_bkg, ew_test_bkg, X_test_sig, ew_test_sig, figure_text=''):
#  plt.hist()


def plotFinalTestOutput(X_test_bkg, ew_test_bkg, X_test_sig, ew_test_sig, figure_text, xlabelText):
  ax = sns.distplot(X_test_bkg, #X_test.query("ylabel==0").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_test_bkg #X_test.query("ylabel==0").loc[:,"eventweight"]
                              }, 
                    color="tab:blue",
                    label="bkg")
  sns.distplot(X_test_sig, #X_test.query("ylabel==1").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_test_sig #X_test.query("ylabel==1").loc[:,"eventweight"]
                              }, 
                    color="tab:orange",
                    label="sig", 
                    ax=ax)
  ax.set(xlabel=xlabelText, ylabel="Entries", yscale="log", title = figure_text)
  sns.set(font= 'stixgeneral')
  #if figure_text:
  #  plt.rc('text', usetex='True')
  #  plt.text(0.5, 0.9, figure_text)
  plt.legend()

  return


def plotFinalTestOutputWithData(X_test_bkg, ew_test_bkg, X_test_sig, ew_test_sig, output_data,ew_data, figure_text, xlabelText):
  ax = sns.distplot(X_test_bkg, #X_test.query("ylabel==0").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_test_bkg #X_test.query("ylabel==0").loc[:,"eventweight"]
                              }, 
                    color="tab:blue",
                    label="bkg")
  
  sns.distplot(X_test_sig, #X_test.query("ylabel==1").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_test_sig #X_test.query("ylabel==1").loc[:,"eventweight"]
                              }, 
                    color="tab:orange",
                    label="sig")
  sns.distplot(output_data, #X_test.query("ylabel==1").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_data #X_test.query("ylabel==1").loc[:,"eventweight"]
                              }, 
                    color="tab:green",
                    label="data", 
                    ax=ax)
  ax.set(xlabel=xlabelText, ylabel="Entries", yscale="log", title = figure_text)
  sns.set(font= 'stixgeneral')
  
  #if figure_text:
  #  plt.rc('text', usetex='True')
  #  plt.text(0.5, 0.9, figure_text)
  plt.legend()

  return


def plotSetup(results_path, X_train, X_test, model):

  plt.figure(1)
  plotTrainTestOutput(X_train.query("ylabel==0").loc[:,"output"], None,
                      X_train.query("ylabel==1").loc[:,"output"], None,
                      X_test.query("ylabel==0").loc[:,"output"], None,
                      X_test.query("ylabel==1").loc[:,"output"], None,
                      'Raw output from training',
                      'XGBoost output')
  plt.savefig(results_path + 'rawPlot.pdf')
  # Plot weighted train and test output, with test set multiplied by 2 to match number of events in training set

            
  # Plot feature importance
  print("model.feature_importances_", model.feature_importances_)
  print("np.sum(model.feature_importances_)", np.sum(model.feature_importances_))
  s_feat_importance = pd.Series(model.feature_importances_, index=X_train.drop(["DatasetNumber", "eventweight", "ylabel", "output", 'RandomRunNumber', 'EventNumber', 'Type'], axis=1).columns)
  print("X_train.drop(['eventweight', 'ylabel'], axis=1).columns\n", X_train.drop(["eventweight", "DatasetNumber", "ylabel", "output", 'RandomRunNumber', 'EventNumber', 'Type'], axis=1).columns)
  
  s_feat_importance.rename({"nBJet20_MV2c10_FixedCutBEff_85" : "b-jets"}, inplace=True)
    
  plt.figure(4)
  sns.set(style="ticks", color_codes=True)
  ax = sns.barplot(x=s_feat_importance*100, y=s_feat_importance.index, ci = None)#, palette="Blues_r")
  #ax.set_yticklabels(s_feat_importance.index)
  ax.set(xlabel="Feature importance [%]")
  sns.set(font= 'stixgeneral')
  plt.savefig(results_path + 'featureImportance.pdf')
  plt.close()
  # Plot ROC curve
  plt.figure(5)
  fpr, tpr, thresholds = metrics.roc_curve(X_test.loc[:,"ylabel"], X_test.loc[:,"output"])
  auc = metrics.roc_auc_score(X_test.loc[:,"ylabel"], X_test.loc[:,"output"])
  ax = sns.lineplot(x=fpr, y=tpr, estimator=None, label='ROC curve: AUC = %0.2f' % auc)
  plt.plot([0,1], [0,1], linestyle="--")
  ax.set(xlabel="False positive rate", ylabel="True positive rate")
  sns.set(font= 'stixgeneral')
  plt.savefig(results_path + 'ROCcurve.pdf')
  plt.close()
  #plt.show()
  return

def plotLoop(results_path, datasetlist, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext):
  plt.figure(datasetlist[i])
  plotTrainTestOutput(X_train.query("ylabel==0").loc[:,"output"],
                      #sum_bkg_scaled_weights_train*X_train.query("ylabel==0").loc[:,"eventweight"],
                      scale_fac*X_train.query("ylabel==0").loc[:,"eventweight"],
                      X_train.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"output"],
                      X_train.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"eventweight"],
                      X_test.query("ylabel==0").loc[:,"output"],
                      #sum_bkg_scaled_weights_train*sum_bkg_scaled_weights*X_test.query("ylabel==0").loc[:,"eventweight"],
                      2.22*X_test.query("ylabel==0").loc[:,"eventweight"],
                      #3*X_test.query("ylabel==0").loc[:,"eventweight"],
                      X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"output"],
                      #sum_sig_scaled_weights*X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"eventweight"],
                      2.22*X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"eventweight"],
                      #3*X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"eventweight"],
                      figuretext, 'XGBoost output')
  plt.savefig(results_path + 'scaled_train_test_'+ datasetlist[i]+'.pdf')
  # Plot final signal vs background estimate for test set, scaled to 10.6/fb
  plt.close()
  #plt.clf()
  plt.figure(datasetlist[i])
  plotFinalTestOutput(X_test.query("ylabel==0").loc[:,"output"],
                      3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                      X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"output"],
                      3.33*X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"eventweight"],
                      figuretext, 'XGBoost output')#,
  #figure_text='10~fb$^{-1}$')
  plt.savefig(results_path + 'test_'+ datasetlist[i]+'.pdf')
  plt.close()
  #plt.cla()

  return



def plotSetupNN(results_path, X_train, X_test, model):

  plt.figure(1)
  plotTrainTestOutput(X_train.query("ylabel==0").loc[:,"output"], None,
                      X_train.query("ylabel==1").loc[:,"output"], None,
                      X_test.query("ylabel==0").loc[:,"output"], None,
                      X_test.query("ylabel==1").loc[:,"output"], None,
                      'Raw output from training', 'Neural network output')
  plt.savefig(results_path + 'rawPlot.pdf')
  # Plot weighted train and test output, with test set multiplied by 2 to match number of events in training set
  plt.close()
  # Plot ROC curve
  plt.figure(5)
  fpr, tpr, thresholds = metrics.roc_curve(X_test.loc[:,"ylabel"], X_test.loc[:,"output"])
  auc = metrics.roc_auc_score(X_test.loc[:,"ylabel"], X_test.loc[:,"output"])
  ax = sns.lineplot(x=fpr, y=tpr, estimator=None, label='ROC curve: AUC = %0.2f' % auc)
  plt.plot([0,1], [0,1], linestyle="--")
  ax.set(xlabel="False positive rate", ylabel="True positive rate")
  plt.savefig(results_path + 'ROCcurve.pdf')
  plt.close()

  #plt.show()
  return

def plotLoopNN(results_path, datasetlist, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext):
  plt.figure(datasetlist[i])
  plotTrainTestOutput(X_train.query("ylabel==0").loc[:,"output"],
                      #sum_bkg_scaled_weights_train*X_train.query("ylabel==0").loc[:,"eventweight"],
                      scale_fac*X_train.query("ylabel==0").loc[:,"eventweight"],
                      X_train.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"output"],
                      X_train.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"eventweight"],
                      X_test.query("ylabel==0").loc[:,"output"],
                      #sum_bkg_scaled_weights_train*sum_bkg_scaled_weights*X_test.query("ylabel==0").loc[:,"eventweight"],
                      2.22*X_test.query("ylabel==0").loc[:,"eventweight"],
                      X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"output"],
                      #sum_sig_scaled_weights*X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"eventweight"],
                      2.22*X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"eventweight"],
                      figuretext, 'Neural network output')
  plt.savefig(results_path + 'scaled_train_test_'+ datasetlist[i]+'.pdf')
  # Plot final signal vs background estimate for test set, scaled to 10.6/fb
  plt.close()
  #plt.clf()
  plt.figure(datasetlist[i])
  plotFinalTestOutput(X_test.query("ylabel==0").loc[:,"output"],
                      3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                      X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"output"],
                      3.33*X_test.query("ylabel==1 & DatasetNumber=="+ datasetlist[i]).loc[:,"eventweight"],
                      figuretext, 'Neural network output')#,
  #figure_text='10~fb$^{-1}$')
  plt.savefig(results_path + 'test_'+ datasetlist[i]+'.pdf')
  plt.close()
  #plt.cla()

  return




