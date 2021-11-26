import pandas as pd
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt

# Построение и оценка модели
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

def merge_dfs(df, data):
    # добавляем столбец с конвертированной датой buy_time
    df['date_t'] = df['buy_time'].apply(lambda x: date.fromtimestamp(x))
    data['date_f'] = data['buy_time'].apply(lambda x: date.fromtimestamp(x))

    data['idx_f'] = data.index
    df['idx'] = df.index

    # объединяем по id
    df_merge = pd.merge(df, data, left_on='id', right_on='id')

    # столбец с разницей по времени
    df_merge['date_dif'] = np.abs(df_merge['date_t']-df_merge['date_f'])

    # группируем и оставляем строки, имеющие минимальную разницу
    res = df_merge.loc[df_merge.groupby(['idx'])['date_dif'].idxmin()]

    # удаляем, переименовываем столбцы
    res.drop(['buy_time_y', 'date_f', 'idx', 'idx_f', 'date_dif'], axis=1, inplace=True)
    res = res.rename(columns={'buy_time_x': 'buy_time', 'date_t': 'date'})
    res = res.reset_index(drop=True)
    return res
    
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')    
    return df

# функция, определяющая тип признака
def features_types(X_train):
    data_train_unique = X_train.nunique()
    # Число константных признаков
    feat_const = set(data_train_unique[data_train_unique == 1].index.tolist())
    len_feat_const = len(feat_const)
    # Число вещественных признаков   
    feat_float = (X_train.fillna(0).astype(int).sum() - X_train.fillna(0).sum()).abs()
    feat_float = set(feat_float[feat_float > 0].index.tolist())
    len_feat_float = len(feat_float)
    # Число целочисленных признаков   
    feat_oth = set(data_train_unique.index.tolist()) - (feat_float | feat_const)
    len_feat_oth = len(feat_oth)   
    # Число категориальных признаков 
    feat_categorical = set(data_train_unique.loc[feat_oth][data_train_unique.loc[feat_oth] <= 10].index.tolist())
    len_feat_categorical = len(feat_categorical)
    
    # update Число целочисленных признаков     
    feat_oth = feat_oth - feat_categorical
    len_feat_oth = len(feat_oth)
    
    feat_float = feat_float | feat_oth
    feat_oth = feat_oth - feat_float
    
    feat_ok = list(feat_categorical | feat_float)
    feat_categorical, feat_float = list(feat_categorical), list(feat_float)
    
    return feat_ok, feat_const, feat_categorical, feat_float


def balance_df_by_target(df, target_name, method='over'):
    assert method in ['over', 'under', 'tomek', 'smote'], 'Неверный метод сэмплирования'
    target_counts = df[target_name].value_counts()

    major_class_name = target_counts.argmax()
    minor_class_name = target_counts.argmin()

    disbalance_coeff = int(target_counts[major_class_name] / target_counts[minor_class_name]) - 1
    if method == 'over':
        for i in range(disbalance_coeff):
            sample = df[df[target_name] == minor_class_name].sample(target_counts[minor_class_name])
            df = df.append(sample, ignore_index=True)
            
    elif method == 'under':
        df_ = df.copy()
        df = df_[df_[target_name] == minor_class_name]
        tmp = df_[df_[target_name] == major_class_name]
        df = df.append(tmp.iloc[
            np.random.randint(0, tmp.shape[0], target_counts[minor_class_name])
        ], ignore_index=True)

    elif method == 'tomek':
        from imblearn.under_sampling import TomekLinks
        tl = TomekLinks()
        X_tomek, y_tomek = tl.fit_sample(df.drop(columns=target_name), df[target_name])
        df = pd.concat([X_tomek, y_tomek], axis=1)
    
    elif method == 'smote':
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy=0.4)
        X_smote, y_smote = smote.fit_resample(df.drop(columns=target_name), df[target_name])
        df = pd.concat([X_smote, y_smote], axis=1)

    return df.sample(frac=1)


def get_classification_report(y_train_true, y_train_pred_proba, y_test_true, y_test_pred_proba, threshold):

    y_train_pred = y_train_pred_proba > threshold
    y_test_pred = y_test_pred_proba > threshold
    print('TRAIN\n\n' + classification_report(y_train_true, y_train_pred))
    print('TEST\n\n' + classification_report(y_test_true, y_test_pred))
    
    
def plot_confusion_matrix(cm, classes,
                          model_name="",
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest',  cmap=cmap)
    
    plt.grid(False)
    plt.title('%s: confusion matrix' % model_name)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    
def plot_roc_curve(fpr, tpr, model_name="", color=None):
    plt.plot(fpr, tpr, label='%s: ROC curve (area = %0.2f)' %
             (model_name, auc(fpr, tpr)), color=color)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0.0, 1.0, 0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s: Receiver operating characteristic curve' % model_name)
    plt.legend(loc="lower right")
    
def plot_precision_recall_curve(recall, precision, model_name="", color=None, max_y=0.2):
    plt.plot(recall, precision, label='%s: Precision-Recall curve (area = %0.2f)' %
             (model_name, auc(recall, precision)), color=color)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("%s: Precision-Recall curve" % model_name)
    plt.axis([0.0, 1.0, 0.0, max_y])
    plt.legend(loc="lower left")
    

def run_grid_search(estimator, X, y, params_grid, cv, scoring='roc_auc'):
    gsc = GridSearchCV(estimator, params_grid, scoring=scoring, cv=cv, n_jobs=-1)

    gsc.fit(X, y)
    print("Best %s score: %.2f" % (scoring, gsc.best_score_))
    print()
    print("Best parameters set found on development set:")
    print()
    print(gsc.best_params_)
    print()
    print("Grid scores on development set:")
    print()

    for i, params in enumerate(gsc.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (gsc.cv_results_['mean_test_score'][i], gsc.cv_results_['std_test_score'][i] * 2, params))

    print()
    
    return gsc

def run_cv(estimator, cv, X, y, scoring='roc_auc', model_name=""):
    cv_res = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    print("%s: %s = %0.2f (+/- %0.2f)" % (model_name,
                                         scoring,
                                         cv_res['test_score'].mean(),
                                         cv_res['test_score'].std() * 2))
    
def prepare_df_test(df, med_1, med_3, med_5, med_207):
    # Удаляем лишние столбцы и генерим новые по дате
    df['month'] = df['date'].map(lambda x: x.month)
    df['day'] = df['date'].map(lambda x: x.day)
    df.drop(['date'], axis=1, inplace=True)
    
    df = df.merge(med_1, how='left', on='vas_id')
    df = df.merge(med_3, how='left', on='vas_id')
    df = df.merge(med_5, how='left', on='vas_id')
    df = df.merge(med_207, how='left', on='vas_id')
    
    print("Добавлены новые фичи")
    df = df.drop(['89', '94', '12', '82','90','22','123','122','105', '109','142', '92', '161', '200',
                            '220', '189', '86', '120', '27', '176', '232', '121', '72', '91', '124', '212', '153',
                            '173', '197', '119', '93', '80', '78', '178', '175', '180', '163', '177', '57', '79', 
                            '118', '88', '17', '199', '216', '218', '221','155', '35', '83', '154', '16', '33', 
                            '85', '95', '31', '202', '179', '32', '81', '84', '139', '75', '23', '24', 
                            '87', '203','15'], axis=1)
    print("Удалены столбцы с низкой значимостью")
    return df

def prepare_df_train(df):
    # Удаляем лишние столбцы и генерим новые по дате
    df['month'] = df['date'].map(lambda x: x.month)
    df['day'] = df['date'].map(lambda x: x.day)
    df.drop(['date'], axis=1, inplace=True)

    med_1 = df.groupby(by='vas_id').agg('1').median().rename('med_1')
    med_3 = df.groupby(by='vas_id').agg('3').median().rename('med_3')
    med_5 = df.groupby(by='vas_id').agg('5').median().rename('med_5')
    med_207 = df.groupby(by='vas_id').agg('207').median().rename('med_201')
    
    df = df.merge(med_1, how='left', on='vas_id')
    df = df.merge(med_3, how='left', on='vas_id')
    df = df.merge(med_5, how='left', on='vas_id')
    df = df.merge(med_207, how='left', on='vas_id')
    
    print("Добавлены новые фичи")
    df = df.drop(['target', 'user_vas', '89', '94', '12', '82','90','22','123','122','105', '109','142', '92', '161', '200',
                            '220', '189', '86', '120', '27', '176', '232', '121', '72', '91', '124', '212', '153',
                            '173', '197', '119', '93', '80', '78', '178', '175', '180', '163', '177', '57', '79', 
                            '118', '88', '17', '199', '216', '218', '221','155', '35', '83', '154', '16', '33', 
                            '85', '95', '31', '202', '179', '32', '81', '84', '139', '75', '23', '24', 
                            '87', '203','15'], axis=1)
    
    print("Удалены столбцы с низкой значимостью")
    
    return df, med_1, med_3, med_5, med_207
 