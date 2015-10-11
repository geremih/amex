import math
import re
import pdb
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier, LinearRegression
import pickle
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, ClassifierMixin
from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import itertools
# from lasagne.layers import DenseLayer
# from lasagne.layers import InputLayer
# from lasagne.layers import DropoutLayer
# from lasagne.nonlinearities import softmax
# from lasagne.updates import nesterov_momentum
# from nolearn.lasagne import NeuralNet
# from sklearn.qda import QDA

# from mlxtend.classifier import EnsembleClassifier

# from hyperopt import hp
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

class MixedClassifier(BaseEstimator, ClassifierMixin):
    
    def fit(self, X, y=None, **fit_args):
        self.feat = SelectKBest(f_classif, k=30).fit(X, y.ravel())
        X_log = self.feat.transform(X)

        print 'Fitting Logistic'
        self.log = LogisticRegression().fit(X_log,y)
        print 'Fitting GBM'
        self.gbm = xgb.XGBClassifier(n_estimators=400, subsample=0.9, colsample_bytree=0.9, learning_rate=0.05, max_depth=4).fit(X,y)
        print 'Done fitting'
        return self

    def predict(self, X):
        prob_log = self.log.predict_proba(self.feat.transform(X))
        prob_gbm = self.gbm.predict_proba(X)
        alpha = .98

        
        prob = (1-alpha)*prob_log + alpha*prob_gbm
        return prob.argmax(axis=1)


class RFTransformer(BaseEstimator, TransformerMixin):


    def fit(self, X, y=None, **fit_args):
        print 'Training RF'
        rf = RandomForestClassifier(n_estimators=300, criterion='gini', max_features=12,bootstrap=True, n_jobs=-1, verbose=True)
        self.classifier =  rf.fit(X,y)
        print 'DONE'
        return self
    
    def transform(self, X):
        return self.classifier.predict_proba(X)

class TreesTransformer(BaseEstimator, TransformerMixin):


    def fit(self, X, y=None, **fit_args):
        self.classifier =   RandomForestClassifier(n_estimators=300, criterion='gini', max_features='auto', bootstrap=False, oob_score=False, n_jobs=-1, verbose=True).fit(X,y)

        return self
    
    def transform(self, X):
        return self.classifier.predict_proba(X)

class GBMTransformer(BaseEstimator, TransformerMixin):


    def fit(self, X, y=None, **fit_args):
        print 'Training GBM'
        params = {'colsample_bytree': 0.6, 'n_estimators': 175.0, 'subsample': 0.6, 'learning_rate': 0.1, 'max_depth':4, 'gamma':.9}
        params['n_estimators'] = int(    params['n_estimators'])
        xg = xgb.XGBClassifier(**params)
        self.classifier = xg.fit(X,y)
        print 'DONE<'
        return self
    
    def transform(self, X):
        return self.classifier.predict_proba(X)
    
class IdempotentTransformer(BaseEstimator, TransformerMixin):


    def fit(self, X, y=None, **fit_args):
        return self
    
    def transform(self, X):
        return X


    
def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    
    Modified from: https://gist.github.com/kljensen/5452382
    
    Details:
    
    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df


def predict_income(df):

    X = df.copy()
    y = X['income'].replace(0, np.nan)

    X['sum'] = X['d_ebony']+ X['d_tokugawa'] + X['d_odyssey'] + X['d_cosmos'] + X['d_centaur']
    X = X[['age', 'education', 'occ', 'h_size', 'married', 'home', 'sum','d_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos']]
    
    one_hot = ['age', 'education','occ']
    #=    df['previous_vote'] = df['previous_vote'].cat.rename_categories(np.arange(5))
    X = encode_onehot(X, one_hot)

    
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X)


    X_train = X[y.notnull()]
    X_test = X[y.isnull()]

    df_train = df[y.notnull()]
    df_test = df[y.isnull()]
    t_train = y[y.notnull()]
    print X_train.shape, t_train.shape
    print 'Training kneighbours'
    neigh = KNeighborsRegressor(n_neighbors=4).fit(X_train, t_train)
    t_out = neigh.predict(X_test)
    print 'Done trainng'
    df_train.loc[:,'income'] = t_train
    df_test.loc[:,'income'] = t_out

    
    return df
    
    # scores = cross_val_score(neigh, X, y, scoring='mean_absolute_error', cv=3, verbose=3)
    # print 'cross_val_score = %s' %(scores)
    # print 'cross val average ', np.mean(scores)

def preprocess(df):




    #    df['previous_vote'] = df['previous_vote'].astype('category')    

    #    df = df.set_index('Citizen_ID')
    for column in ['d_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos']:
        df[column] = df[column].str.rstrip().str.replace(',','').str.replace('$','').astype(int)
    #Convert to  string as DictVectorizer requires strings
    df['region'] = df['region'].astype(str)

    #    df = predict_income(df)
    #randomize and predict would set stuff in sync
    #df = df.fillna(0)

    one_hot = [ 'previous_vote', 'age', 'occ', 'education', 'region']
    #=    df['previous_vote'] = df['previous_vote'].cat.rename_categories(np.arange(5))
    df = encode_onehot(df, one_hot)

    df['income'] = df['income'].replace(0,np.nan)
    df = df.fillna(0)
#     rallies = ['n_centaur', 'n_ebony', 'n_tokugawa', 'n_odyssey', 'n_cosmos']
#     df.loc[:, rallies] = df[rallies].fillna(0)
#     n_sum = df['n_ebony']+ df['n_tokugawa'] + df['n_odyssey'] + df['n_cosmos']
# #    df['n_centaur_wrong'] = (df['n_rallies'] != (n_sum + df['n_centaur'])).astype(int)
#     df['n_centaur'] = df['n_rallies']  - n_sum

#     #Increased accuracy
#     df['d_sum'] = df['d_ebony']+ df['d_tokugawa'] + df['d_odyssey'] + df['d_cosmos'] + df['d_centaur']

    #Decreased accuracy
    #    df['s_sum'] = df['s_ebony']+ df['s_tokugawa'] + df['s_odyssey'] + df['s_cosmos'] + df['s_centaur']
    


    parties = ['centaur', 'cosmos', 'odyssey', 'tokugawa', 'ebony']
    
    df = df.fillna(0)
    def row_greater(col1, col2):

        def fn(row):
            return row[col1] - row[col2]
        return fn
    for first, second in itertools.combinations(parties, 2):
        #only this got 651500
        df['n_rel' + first + second] = df.apply(row_greater('n_'+first, 'n_'+second), axis=1)

        
    df = drop_features(df)
    return df

def get_previous(instance,categories):
    for i, c in enumerate(categories):
        if instance['previous_vote='+c]:
            return i


def drange(start, stop, step):
     r = start
     while r < stop:
     	 yield r
     	 r += step        
    
def get_predicted(probs, alpha, instance,categories):
    max_index = -1
    max_prob = 0
    previous = get_previous(instance,categories)
    for i,p in enumerate(probs):
        if max_prob < p:
            max_index = i
            max_prob = p
    probs = sorted(zip(probs, np.arange(0,5)))
    if probs[-1][1] == previous:
        if probs[-1][0] * alpha > probs[-2][0]:
            return previous
        else:
            
            return probs[-2][1]
    else:
        return probs[-1][1]
    
    


def testing(model, X,ids,categories):
    t = model.predict_proba(X)
    t[:,3] = 0
    t = t.argmax(axis=1)
    t = [categories[i] for i in t]
    pd.DataFrame({'y': t},index=ids).to_csv('alpha_kappa_cow_iit_kgp_1.csv')

def metric(X,y):
    # evaluate the model by splitting into train and test sets
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # model2 = get_model()
    # model2.fit(X_train, y_train)

    # # predict class labels for the test set
    # predicted = model2.predict(X_test)
    # print 'predicted = %s' %(predicted)

    # generate evaluation metrics
 #   print 'metrics.accuracy_score(y_test, predicted) = %s' %(metrics.accuracy_score(y_test, predicted))
    #evaluate the model using 10-fold cross-validation
    scores = cross_val_score(get_model(), X, y, scoring='accuracy', cv=3, verbose=3)
    print 'cross_val_score = %s' %(scores)
    print 'cross val average ', np.mean(scores)
    return np.mean(scores)

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')
    
def drop_features(df)    :
    del df['Citizen_ID']
    #Increased accuracy
    del df['docs']
    return df

def get_features(df) :
    features =  ['region=8.0', 'income=0','region=3.0','region=6.0','n_ebony','income','education=primary','education=MBA','age=18-24','age=55+','age=35-45','age=45-55','age=25-35','home','married','s_tokugawa','d_tokugawa','s_odyssey','d_odyssey','s_centaur','d_centaur','h_size','n_unique_p','d_ebony','d_cosmos','politics','s_ebony','p_voted','s_cosmos','r_years','n_tokugawa','n_rallies','previous_vote=TOKUGAWA','n_centaur','n_odyssey','previous_vote=EBONY','n_cosmos','previous_vote=ODYSSEY','previous_vote=CENTAUR','previous_vote=COSMOS']
    return df[features]

def create_nn(X):
    
    layers0 = [('input', InputLayer),
               ('dense0', DenseLayer),
               ('dropout', DropoutLayer),
               ('dense1', DenseLayer),
               ('output', DenseLayer)]
    num_classes = 5
    num_features = X.shape[1]    
    net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=200,
                 dropout_p=0.5,
                 dense1_num_units=200,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 
                 eval_size=0.2,
                 verbose=1)

    return net0
def get_model():

    #return LogisticRegression()
#    return svm.SVC()
#    return GradientBoostingClassifier(n_estimators=200, max_depth=4,learning_rate=.05)
    #best has occured at 0.05 and 400
#    return

    #650200
#    params = {'n_estimators': 549.0, 'subsample': 0.55, 'colsample_bytree': 0.8500000000000001, 'gamma': 0.7000000000000001, 'learning_rate': 0.025, 'max_depth': 5.0, 'min_child_weight': 1.0}

    #649500
#    params = {'n_estimators': 585.0, 'subsample': 0.9, 'colsample_bytree': 0.55, 'gamma': 0.75, 'learning_rate': 0.025, 'max_depth': 5.0, 'min_child_weight': 4.0}
#    params = {'n_estimators': 534.0, 'subsample': 0.6000000000000001, 'colsample_bytree': 0.65, 'gamma': 0.75, 'learning_rate': 0.025, 'max_depth': 4.0, 'min_child_weight': 1.0}
#650700
    #650700
#    params = {'colsample_bytree': 0.75, 'min_child_weight': 1.0, 'n_estimators': 380.0, 'subsample': 0.55, 'learning_rate': 0.025, 'max_depth': 6.0, 'gamma': 0.6000000000000001, 'seed': np.random.random()}

    params = {'colsample_bytree': 0.6, 'n_estimators': 175.0, 'subsample': 0.6, 'learning_rate': 0.1, 'max_depth':4, 'gamma':.9}
    params['n_estimators'] = int(    params['n_estimators'])
    xg = xgb.XGBClassifier(**params)
    rf = RandomForestClassifier(n_estimators=1500, criterion='gini', max_features=25,bootstrap=True, n_jobs=-1, verbose=True)
    lr = LogisticRegression()
    knn = KNeighborsClassifier(n_neighbors=40)
    
    bag =  BaggingClassifier(xg, max_samples=.5, n_estimators=250, verbose=3)
    ada = AdaBoostClassifier(xg)
    return bag

    #combined_features = FeatureUnion([ ('RF', RFTransformer()), ('gt', GBMTransformer())])
    #return Pipeline([ ('combined', combined_features),('ridge',RidgeClassifier(alpha=750))])
#    return EnsembleClassifier([xg,rf], verbose=3, voting='soft')
#    return create_nn(X) 
#    return LogisticRegression()
#    return MixedClassifier()
#    return RidgeClassifier()
#    return AdaBoostClassifier(xgb.XGBClassifier(n_estimators=400, subsample=0.9, colsample_bytree=0.9, learning_rate=0.05, max_depth=6)
#    return AdaBoostClassifier()

#    return 
#    return BaggingClassifier(RandomForestClassifier(n_estimators=300, criterion='gini', max_features='auto', bootstrap=False, oob_score=False, n_jobs=-1, verbose=True)    ,verbose=3)

#    return RandomForestClassifier(n_estimators=200)
#    return ExtraTreesClassifier()


def grid_search(X,y):
    # parameters = {
    #     'gamma': np.arange(0,.2,.03)
    # }
    parameters = {
        'alpha': np.arange(100,1001,100)
    }    
    model = GridSearchCV(get_get_model(), parameters, verbose=3)
    model.fit(X,y)
    best_parameters, score, _ = max(model.grid_scores_, key=lambda x: x[1])
    print(score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

def random_search(X,y):
    parameters = {'gamma':uniform(loc=0,scale=0.99)}
    model = RandomizedSearchCV(get_model(), parameters, verbose=3)
    model.fit(X,y)
    print(model.best_score_)
    print(model.best_params_)
    print(model.best_estimator_)
    best_parameters, score, _ = max(model.grid_scores_, key=lambda x: x[1])
    print(score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

def reconsitute_previous(X, categories):
    X['previous_vote'] = 0
    for i, c in enumerate(categories):
        X.loc[X['previousvote'+c] == 1,['previous_vote']] = i
    return X

def read_data():
    df = pd.read_csv('training.csv')
        #Set columns to readable names
    df.columns = ['Citizen_ID', 'actual_vote', 'previous_vote', 'd_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos', 's_centaur', 's_ebony', 's_tokugawa', 's_cosmos', 's_odyssey', 'occ', 'region', 'h_size', 'age', 'married', 'home', 'politics', 'r_years', 'p_voted', 'n_unique_p', 'education', 'n_centaur', 'n_ebony', 'n_tokugawa', 'n_odyssey', 'n_rallies', 'n_cosmos', 'docs', 'income']
    df['actual_vote'] = df['actual_vote'].astype('category')
    categories = df['actual_vote'].cat.categories
    print categories
    df['actual_vote'] = df['actual_vote'].cat.rename_categories(np.arange(5))
    y = df['actual_vote']
    del df['actual_vote']

    tf = pd.read_csv('final.csv')
    tf.columns = ['Citizen_ID', 'previous_vote', 'd_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos', 's_centaur', 's_ebony', 's_tokugawa', 's_cosmos', 's_odyssey', 'occ', 'region', 'h_size', 'age', 'married', 'home', 'politics', 'r_years', 'p_voted', 'n_unique_p', 'education', 'n_centaur', 'n_ebony', 'n_tokugawa', 'n_odyssey', 'n_rallies', 'n_cosmos', 'docs', 'income']
    len_df = len(df)
    len_tf = len(tf)
    ids = tf['Citizen_ID']
    df = pd.concat([df, tf],ignore_index = True)
    X = preprocess(df)

    def remove_punctuation(string):
        return re.sub('[^0-9a-zA-Z]+', '', string)
    X.columns = map(remove_punctuation,X.columns)
    print X.columns
    X_test = X[len_df:]
    X = X[:len_df]
    print 'X_test.shape = %s  ids.shape = %s' %(X_test.shape,ids.shape)
    print 'X.shape = %s  y.shape = %s' %(X.shape,y.shape)


    return X, y, X_test, ids, categories




def hyper_optimize(X,y):
    space = {
             'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
             'learning_rate' : hp.quniform('eta', 0.025, 0.5, 0.025),
             'max_depth' : hp.quniform('max_depth', 1, 13, 1),
             'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
             'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
             'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
             }
    trials = Trials()    

    def score(params):
        print "Training with params : "
        print params
        
        params['n_estimators'] = int(params['n_estimators'])
        model = xgb.XGBClassifier(**params)
        print model
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=3, verbose=3)
        score = -1 * np.mean(scores)
        print "\tScore {0}\n\n".format(score)
        return {'loss': score, 'status': STATUS_OK}    

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)

    print best
    return trials

def hyper_optimize_rf(X,y):
    space = {
             'max_features' : hp.quniform('max_features', 1, 40, 1),
             'n_estimators' : hp.quniform('n_estimators', 300, 2000, 100),
             'criterion' : hp.choice('criteria',["gini", "entropy"]),
        'bootstrap': hp.choice('boostrap',[True, False])
             }
    trials = Trials()    

    def score(params):
        print "Training with params : "
        print params
        params['max_features'] = int(params['max_features'])
        params['n_estimators'] = int(params['n_estimators'])
        model = RandomForestClassifier(**params)
        print model
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=3, verbose=3,n_jobs=-1)
        score = -1 * np.mean(scores)
        print "\tScore {0}\n\n".format(score)
        return {'loss': score, 'status': STATUS_OK}    

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)

    print best
    return trials



def main():
    X, y, X_test, ids, categories = read_data()
#    feat = SelectKBest(f_classif, k=100).fit(X, y.ravel())
#    X = feat.transform(X)
#    X_test = feat.transform(X_test)
    #feat = PCA(n_components = 50).fit(X)
    #X = feat.transform(X)
    #X_test = feat.transform(X_test)

    #    X = X.values.copy()
    #    y = pd.DataFrame(y).values.copy()[:,-1 ]
    print y.shape
    model = get_model()
    print 'Training model'
    model = model.fit(X,y)
    print 'Done Training'
    print model.score(X, y)
#    xgb.plot_importance(model.booster())
    # print model.featuremportances_
    # print len(model.feature_importances_)
    # print sorted(zip(model.feature_importances_, X.columns.ravel()))


    #    hyper_optimize_rf(X,y)
#    grid_search(X,y)
    #    random_search(X,y)
    testing(model, X_test,ids, categories)
    #bst = model.booster()
    #bst.save_model('250.model')
    #bst.dump_model('dump.raw.txt','featmap.txt')


    metric(X,y)

    
if __name__ == '__main__':
    main()




    
