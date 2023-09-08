import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
test_data=pd.read_csv()

with open('MLP_model.pkl', 'rb') as f:
    MLP_model = pickle.load(f)

with open('XGB2model.pkl','rb') as f:
    XGB_model =pickle.load(f)

with open('SVMmodel.pkl','rb') as f:
    SVM_model=pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)



with open('Hybrid_XGB.pkl','rb') as f:
    Hybrid_XGB=pickle.load(f)


X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

X_test=X_test['text'].squeeze()
X_test= vectorizer.transform(X_test.values.astype('U'))

MLP_pred = MLP_model.predict(X_test)
MLP_fpr, MLP_tpr,MLP_thresholds = roc_curve(y_test, MLP_pred)
MLP_roc_auc = auc(MLP_fpr, MLP_tpr)
MLP_auc_roc = roc_auc_score(y_test, MLP_pred)

XGB_pred = XGB_model.predict(X_test)
XGB_fpr, XGB_tpr,XGB_thresholds = roc_curve(y_test, XGB_pred)
XGB_roc_auc = auc(XGB_fpr, XGB_tpr)
XGB_auc_roc = roc_auc_score(y_test, XGB_pred)

SVM_pred = SVM_model.predict(X_test)
SVM_fpr, SVM_tpr,SVM_thresholds = roc_curve(y_test, SVM_pred)
SVM_roc_auc = auc(SVM_fpr, SVM_tpr)
SVM_auc_roc = roc_auc_score(y_test, SVM_pred)


Hybrid_prob_pred=MLP_model.predict_proba(X_test)
test_transformed = xgb.DMatrix(Hybrid_prob_pred, label=y_test)
Hybrid_pred = Hybrid_XGB.predict(test_transformed)
Hybrid_pred = np.round(Hybrid_pred)
Hybrid_fpr, Hybrid_tpr,Hybrid_thresholds = roc_curve(y_test, Hybrid_pred)
Hybrid_roc_auc = auc(Hybrid_fpr, Hybrid_tpr)
Hybrid_auc_roc = roc_auc_score(y_test, Hybrid_pred)

plt.figure()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('\'Functioned\' ROC Curve')
plt.plot(MLP_fpr, MLP_tpr, color='orange', lw=2 ,label='MLP classifier, AUC: '+str(round(MLP_auc_roc,3)))
plt.plot(XGB_fpr, XGB_tpr, color='purple', lw=2 ,label='XGBoost classifier, AUC: '+str(round(XGB_auc_roc,3)),linestyle='--')
plt.plot(SVM_fpr, SVM_tpr, color='red', lw=2 ,label='SVM classifier, AUC: '+str(round(SVM_auc_roc,3)))
plt.plot(Hybrid_fpr, Hybrid_tpr, color='yellow', lw=2 ,label='Hybrid classifier, AUC: '+str(round(Hybrid_auc_roc,3)),linestyle='--')
plt.plot([0, 1], [0, 1], color='green', lw=2,label="Random Classifier", linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.show()

