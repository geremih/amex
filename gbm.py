import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import xgboost as xgb

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
    #Set columns to readable names
    df.columns = ['Citizen_ID', 'actual_vote', 'previous_vote', 'd_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos', 's_centaur', 's_ebony', 's_tokugawa', 's_cosmos', 's_odyssey', 'occ', 'region', 'h_size', 'age', 'married', 'home', 'politics', 'r_years', 'p_voted', 'n_unique_p', 'education', 'n_centaur', 'n_ebony', 'n_tokugawa', 'n_odyssey', 'n_rallies', 'n_cosmos', 'docs', 'income']

    df = df.fillna(0)

    #Remove $ and ,
    df = df.set_index('Citizen_ID')
    for column in ['d_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos']:
        df[column] = df[column].str.rstrip().str.replace(',','').str.replace('$','').astype(int)
    #Convert to  string as DictVectorizer requires strings
    df['region'] = df['region'].astype(str)
    one_hot = ['age', 'previous_vote', 'education', 'region', 'occ']
    df = encode_onehot(df, one_hot)
    df['actual_vote'] = df['actual_vote'].astype('category')

    return df

def get_model():
#    return LogisticRegression()
#    return svm.SVC()
#    return GradientBoostingClassifier(n_estimators=200, max_depth=4)
    return xgb.XGBClassifier(max_depth = 5, n_estimators=400, learning_rate=0.05)


def testing(model):
    tf = pd.read_csv('testing.csv')
    tf = clean_testing(tf)
    X = get_features(tf)
    X = fix_data(X)
    t = model.predict(X)
    ids = tf['Citizen_ID']
    t = [categories[i] for i in t]
    pd.DataFrame({'y': t},index=ids).to_csv('alpha_kappa_iit_kgp_3.csv')
def fix_data(X):

    #making columns strictly alphanumeric
    oldcols = X.columns
    newcols = []
    for col in oldcols:
        newcols.append(col.replace('_',''))
    X.columns = newcols
    return X

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


df = pd.read_csv('training.csv')
df = preprocess(df)

categories = df['actual_vote'].cat.categories
df['actual_vote'] = df['actual_vote'].cat.rename_categories(np.arange(5))


y = df['actual_vote']

X = fix_data(df)
print X.columns


print 'X.shape = %s  y.shape = %s' %(X.shape,y.shape)
#model = LogisticRegression()
model = get_model()


model = model.fit(X, y)
print model.score(X, y)

metric(X,y)


