#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## Charts visualisations for partial pependence analytics (PDP) 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
from tqdm import tqdm

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## Target metric count for objects belongs to pair of features and they intervals 
def targetMetric(_df, _fun='mean', colLim={0: [0,0], 1:[0,0]}):
  '''
  InPuts:
    _df - dataSet in pandas DataFrame 
    _fun - name of agregate action for target values ('mean', 'median', 'classification')
    colLim - dict of pairs: 
        key: columnname from _df 
        value: list of left and right values of interested interval
  OutPuts:
   function return aggregate of target values of objects corresponded to conditions of intervals per each interesting columns
  '''
   ## chose objects inside of feture intervals
    _cond = None
    for _key in colLim:
        if not (_cond is None):
            _cond = (_cond)  & (_df[_key] >= colLim[_key][0])  &  (_df[_key]  <= colLim[_key][1])
        else:
            _cond = (_df[_key]  >= colLim[_key][0])  &  (_df[_key]  <= colLim[_key][1])
    _df_filter = _df[_cond]
    # choose objects coresponds in interval
    inIntervalSet = _df_filter['FLAG']

    if optimize_objective == 'mean':
        res = np.nanmean(inIntervalSet)
    elif optimize_objective == 'median':
        res = np.nanmedian(inIntervalSet)
    elif optimize_objective == 'classification':
        if np.size(inIntervalSet, 0) != 0 :
            res = np.sum(inIntervalSet)/np.size(inIntervalSet, 0)
        else: 
            res = 1
    else:
        res = np.nanmean(inIntervalSet)
    return round(res,3)

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Подготовка данных для графиков парного влияния параметров на брак
#2D heatmap
# сформируем данные для отображения карты в 2 и 3 мерном пространстве

# Создаем массивы NumPy с координатами точек по осям X и У. 
# Используем метод meshgrid, при котором по векторам координат 
# создается матрица координат. Задаем нужную функцию Z(x, y).

def gridConstruct(_dt, _colLim):
#   colLim - dictionary: 
#       key - column name in _dt, 
#       value - ndarray, 
#       value[0] means min edge,   
#       value[1] means max edge,   
#       value[2] means step for mesh gfrid   
    _grid = []
    _keys = []  ##auxiliary list to remember column names
    for _key in _colLim:
        _keys.append(_key)
        X = np.arange(_colLim[_key][0],  _colLim[_key][1], _colLim[_key][2])
        _grid.append(X)
    xlen = len(_grid[0])
    ylen = len(_grid[1])

    Z = np.zeros((ylen,xlen))
    _gridLim = _colLim.copy()
    for i in tqdm(range(ylen)[:-1]):
        for k in range(xlen)[:-1]:
            _gridLim = {_keys[1]:[_grid[1][i], _grid[1][i+1]], _keys[0]:[_grid[0][k], _grid[0][k+1]]}
            Z[i,k] = targetMetric(_dt, 'classification', _gridLim)
    X, Y = np.meshgrid(_grid[0], _grid[1])
    return X,Y,Z

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def d3ChartShow(X,Y,Z, _cols):
    # 3D-график 
    plt.figure(figsize=(14,10))
    surf2d  = plt.contourf(X, Y, Z, 8, cmap='BuPu', grid=True, alpha=0.7)
    plt.colorbar(surf2d)
    plt.xlabel(_cols[0])
    plt.ylabel(_cols[1])
    plt.title(_cols[2])
    plt.show()
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////