#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#  SMART OFFERING 
#  DATA PREPARATION UTILS
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## Подготовка данных
def LeadsPreparation(_leads, _agree=None):
    ## Удаляем неселективный признак
    _leads.drop(columns=['CONTENT'], inplace=True)

    numCols = [_leads.columns[2]] + list(_leads.columns[-12:])
    numCols = ['LIFE_TIME']+numCols
    _leads['LIFE_TIME'] = _leads['LIFE_TIME'].astype('str')
    _leads[numCols] = _leads[numCols].apply(lambda x: x.str.replace(',','.'))
    _leads[numCols] = _leads[numCols].apply(lambda x: x.str.replace("[? ]","0", regex=True))
    _leads[numCols] = _leads[numCols].astype('float')

    _leads['CURRENT_AGE'] = _leads['CURRENT_AGE'].astype('int')
    _leads['LIFE_TIME'] = _leads['LIFE_TIME'].astype('int')

    catCols = _leads.columns[3:10]
    _leads[catCols] = _leads[catCols].astype('category')
    ## Крайние 8 столбцов принудительно приводятся к типу float 
    #_leads[_leads.columns[-8:]].astype('float').dtypes

    #_leads[numCols].astype('float').dtypes

    ## Избавляемся от NULL-значений
    _leads[numCols] = _leads[numCols].fillna(0)

    ## Добавляем целевую переменную
    _leads['FLAG'] = 1
    _leads['FLAG_color'] = 'blue'
    if(_agree is not None):
        condition =  _leads['MSISDN'].isin(_agree['MSISDN'])
        idx = _leads.index[~condition]
        _leads.loc[idx,'FLAG'] = 0
    ## Для визуализации добавляем цветовую метку, согласно отклику
        _leads.loc[idx,'FLAG_color'] = 'green'
    return _leads, numCols, catCols
#del agree
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
from sklearn import preprocessing

def encodingCatCols(leads, catCols):
    le_CS = preprocessing.LabelEncoder()

    codeCols = []
    for col in catCols:
        codeCols.append(col+'_encoded')
        leads[col+'_encoded'] = le_CS.fit_transform(leads[col])
    return leads, codeCols
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
from sklearn.preprocessing import StandardScaler, MinMaxScaler
def scalingNumCols(leads, numCols):
    scaler = MinMaxScaler()
    scalCols = []
    for col in numCols:
        scalCols.append(col+'_scaled')
        leads[col+'_scaled'] = scaler.fit_transform(leads[col].to_numpy().reshape(-1, 1))
    return leads, scalCols
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## Clusterization
from sklearn.cluster import KMeans
def AddClusterField(_dt, _n_clusters, ATTR_NAMES):
#
#    _dt - pd.DataFrame,  consists of dimension fields only, index field means unique key
#    _n_clusters - int, number of clusters  
#
    if('class' in _dt):
        _dt.drop('class', axis='columns', inplace=True)
    
    kmeans = KMeans(n_clusters=_n_clusters) # You want cluster observation records into _n_clusters
    kmeans.fit(_dt[ATTR_NAMES])
    
    _dt['class'] = kmeans.predict(_dt[ATTR_NAMES])
    return _dt
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////