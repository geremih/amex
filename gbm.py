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
from sklearn.linear_model import RidgeClassifier
import pickle
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


def preprocess(df):

    df['income=na'] = (df['income']==np.nan).astype(int)

    df = df.fillna(0)

#    df = df.set_index('Citizen_ID')
    for column in ['d_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos']:
        df[column] = df[column].str.rstrip().str.replace(',','').str.replace('$','').astype(int)
    #Convert to  string as DictVectorizer requires strings
    df['region'] = df['region'].astype(str)
    one_hot = ['age', 'previous_vote', 'education', 'region','occ']
    df = encode_onehot(df, one_hot)


    #Remove $ and ,    
    return df

def get_model():
#    return LogisticRegression()
#    return svm.SVC()
#    return GradientBoostingClassifier(n_estimators=200, max_depth=4,learning_rate=.05)
    #best has occured at 0.05 and 400
    return xgb.XGBClassifier(learning_rate=.05, n_estimators=400, max_depth=4, subsample=.9)
#    return RandomForestClassifier(n_estimators=200)
#    return ExtraTreesClassifier()



def testing(model, X,ids):
    t = model.predict(X)
    t = [categories[i] for i in t]
    pd.DataFrame({'y': t},index=ids).to_csv('alpha_kappa_iit_kgp_3.csv')

def metric(X,y):
    # evaluate the model by splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model2 = get_model()
    model2.fit(X_train, y_train)

    # predict class labels for the test set
    predicted = model2.predict(X_test)
    print 'predicted = %s' %(predicted)

    # generate evaluation metrics
    print 'metrics.accuracy_score(y_test, predicted) = %s' %(metrics.accuracy_score(y_test, predicted))
    #evaluate the model using 10-fold cross-validation
    scores = cross_val_score(get_model(), X, y, scoring='accuracy', cv=3)
    print 'cross_val_score = %s' %(scores)
    print 'cross val average ', np.mean(scores)


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')
    
def drop_features(df)    :
    del df['Citizen_ID']
    return df

def get_features(df) :
    features =  ['region=8.0', 'income=0','region=3.0','region=6.0','n_ebony','income','education=primary','education=MBA','age=18-24','age=55+','age=35-45','age=45-55','age=25-35','home','married','s_tokugawa','d_tokugawa','s_odyssey','d_odyssey','s_centaur','d_centaur','h_size','n_unique_p','d_ebony','d_cosmos','politics','s_ebony','p_voted','s_cosmos','r_years','n_tokugawa','n_rallies','previous_vote=TOKUGAWA','n_centaur','n_odyssey','previous_vote=EBONY','n_cosmos','previous_vote=ODYSSEY','previous_vote=CENTAUR','previous_vote=COSMOS']
    return df[features]

df = pd.read_csv('training.csv')
    #Set columns to readable names
df.columns = ['Citizen_ID', 'actual_vote', 'previous_vote', 'd_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos', 's_centaur', 's_ebony', 's_tokugawa', 's_cosmos', 's_odyssey', 'occ', 'region', 'h_size', 'age', 'married', 'home', 'politics', 'r_years', 'p_voted', 'n_unique_p', 'education', 'n_centaur', 'n_ebony', 'n_tokugawa', 'n_odyssey', 'n_rallies', 'n_cosmos', 'docs', 'income']

df['actual_vote'] = df['actual_vote'].astype('category')
categories = df['actual_vote'].cat.categories
df['actual_vote'] = df['actual_vote'].cat.rename_categories(np.arange(5))
y = df['actual_vote']
del df['actual_vote']

tf = pd.read_csv('testing.csv')
tf.columns = ['Citizen_ID', 'previous_vote', 'd_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos', 's_centaur', 's_ebony', 's_tokugawa', 's_cosmos', 's_odyssey', 'occ', 'region', 'h_size', 'age', 'married', 'home', 'politics', 'r_years', 'p_voted', 'n_unique_p', 'education', 'n_centaur', 'n_ebony', 'n_tokugawa', 'n_odyssey', 'n_rallies', 'n_cosmos', 'docs', 'income']
len_df = len(df)
len_tf = len(tf)
ids = tf['Citizen_ID']
df = pd.concat([df, tf],ignore_index = True)
df = preprocess(df)
X = drop_features(df)
X_test = X[len_df:]
X = X[:len_df]
print 'X_test.shape = %s  ids.shape = %s' %(X_test.shape,ids.shape)
print 'X.shape = %s  y.shape = %s' %(X.shape,y.shape)
# feat = SelectKBest(f_classif, k=40).fit(X, y.ravel())
# X = feat.transform(X)
# X_test = feat.transform(X_test)
#feat = PCA().fit(X)
#X = feat.transform(X)
#X_test = feat.transform(X_test)

y = y.ravel()
model = get_model()
model = model.fit(X, y)
print model.score(X, y)

# print model.feature_importances_
# print len(model.feature_importances_)
# print sorted(zip(model.feature_importances_, X.columns.ravel()))

metric(X,y)
testing(model, X_test,ids)

