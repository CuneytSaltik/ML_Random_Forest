import re
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score,f1_score,fbeta_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

dfread = pd.read_csv(r"/Users/cunsal/Documents/DataThesis/CIC/CIC_cleaned.csv", header=0)

df = dfread[dfread['label'] == 1.0].sample(2000)
dfzero = dfread[dfread['label'] == 0.0].sample(2000)
print(df.count())
print(dfzero.count())
df = pd.concat([df, dfzero], ignore_index=True)
print(df.columns)


regex = re.compile(r"\[|\]|<", re.IGNORECASE)
df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]

df = pd.get_dummies(df, columns =['bidirectional_first_seen_ms', 'bidirectional_last_seen_ms', 'bidirectional_duration_ms',
                                  'bidirectional_packets', 'bidirectional_bytes', 'src2dst_first_seen_ms',
                                  'src2dst_last_seen_ms', 'src2dst_duration_ms', 'src2dst_packets', 'src2dst_bytes',
                                  'dst2src_first_seen_ms', 'dst2src_last_seen_ms', 'dst2src_duration_ms', 'dst2src_packets',
                                  'dst2src_bytes', 'bidirectional_min_ps', 'bidirectional_mean_ps', 'bidirectional_stddev_ps',
                                  'bidirectional_max_ps', 'src2dst_min_ps', 'src2dst_mean_ps', 'src2dst_stddev_ps',
                                  'src2dst_max_ps', 'dst2src_min_ps', 'dst2src_mean_ps', 'dst2src_stddev_ps',
                                  'dst2src_max_ps', 'bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms',
                                  'bidirectional_stddev_piat_ms', 'bidirectional_max_piat_ms', 'src2dst_min_piat_ms',
                                  'src2dst_mean_piat_ms', 'src2dst_stddev_piat_ms', 'src2dst_max_piat_ms',
                                  'dst2src_min_piat_ms', 'dst2src_mean_piat_ms', 'dst2src_stddev_piat_ms',
                                  'dst2src_max_piat_ms', 'bidirectional_syn_packets', 'bidirectional_ack_packets',
                                  'bidirectional_psh_packets', 'bidirectional_rst_packets', 'bidirectional_fin_packets',
                                  'src2dst_syn_packets', 'src2dst_ack_packets', 'src2dst_psh_packets', 'src2dst_rst_packets',
                                  'src2dst_fin_packets', 'dst2src_syn_packets', 'dst2src_ack_packets', 'dst2src_psh_packets',
                                  'dst2src_rst_packets', 'dst2src_fin_packets', 'protocol_6', 'protocol_17', 'protocol_58'],
                    drop_first=True)
print(df.columns)
X = df.loc[:, df.columns != "label"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.21, random_state = 0)

#Ignore all warning messages
warnings.filterwarnings("ignore")

rf_clf = RandomForestClassifier (random_state = 42)

params_grid = {
               'max_depth': [5, 8, 10, 15, 20, 25, 30, 50],
               'max_features': ['log2','sqrt',0.25,0.5, .66, 1.0],
                'min_samples_leaf': [1,25,50,70],
               'min_samples_split': [2, 5, 10, 20]
               }

grid_search = GridSearchCV(estimator = rf_clf, param_grid = params_grid ,n_jobs = -1, cv = 2, scoring = 'accuracy')
grid_result = grid_search.fit(X_train, y_train)

gbc_clf2 = RandomForestClassifier(#nthread = grid_result.best_params_.get('nthread'),
                     max_depth = grid_result.best_params_.get('max_depth'),
                     max_features = grid_result.best_params_.get('max_features'),
                     min_samples_leaf = grid_result.best_params_.get('min_samples_leaf'),
                     min_samples_split = grid_result.best_params_.get('min_samples_split')
                      )

gbc_clf2.fit(X_train, y_train)

with open('model_pickle_cic', 'wb') as f:pickle.dump(gbc_clf2, f)

acc_train = accuracy_score(y_train, gbc_clf2.predict(X_train)) * 100
acc_test = accuracy_score(y_test, gbc_clf2.predict(X_test)) * 100
print("accuracy of train phase is {:.4f}".format(acc_train))
print("accuracy of test phase is {:.4f}".format(acc_test))

y_train_pred = gbc_clf2.predict(X_train)
y_test_pred = gbc_clf2.predict(X_test)
print("Mean Squre Error - train {:.4f}".format(mean_squared_error(y_train,y_train_pred)))
print("Mean Squre Error - test {:.4f}".format(mean_squared_error(y_test,y_test_pred)))

plot_confusion_matrix(gbc_clf2, X_test, y_test)
plt.show()
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

print("-------------------------------------Metrics------------------------------------------")
print("Test accuracy score {:.4f}".format(accuracy_score(y_test, y_test_pred)*100))
print("Test Recall {:.4f}".format(recall_score(y_test, y_test_pred)*100))
print("Test Precision {:.4f}".format(precision_score(y_test, y_test_pred)*100))
print("Test F1 Score {:.4f}".format(f1_score(y_test, y_test_pred)*100))
print("Test F2 Score {:.4f}".format(fbeta_score(y_test, y_test_pred, beta=2.0)*100))

print("--------------------------TPR, TNR, FPR, FNR------------------------------------------")
TPR = tp/(tp+fn)
TNR = tn/(tn+fp)
FPR = fp/(fp+tn)
FNR = fn/(fn+tp)
print("TPR {:.4f}".format(TPR))
print("TNR {:.4f}".format(TNR))
print("FPR {:.4f}".format(FPR))
print("FNR {:.4f}".format(FNR))
