#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#  SMART OFFERING 
#  DATA REPORT AND VISUALISATION UTILS
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## Отчеты по итогам обучения
import seaborn as sns
import pandas as pd
from scipy import stats

import time
from IPython import display
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////э
# PDP https://www.kaggle.com/dansbecker/partial-plots-daily

# Correlation heatmap tricks 
#https://stackoverflow.com/questions/39409866/correlation-heatmap
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def HeatMapDataFrame(_df):
    cmap=sns.diverging_palette(5, 250, as_cmap=True)

    def magnify():
        return [dict(selector="th",
                     props=[("font-size", "7pt")]),
                dict(selector="td",
                     props=[('padding', "0em 0em")]),
                dict(selector="th:hover",
                     props=[("font-size", "12pt")]),
                dict(selector="tr:hover td:hover",
                     props=[('max-width', '200px'),
                            ('font-size', '12pt')])
    ]
    return _df.style.background_gradient(cmap, axis=1)\
        .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
        .set_caption("Q matrix")\
        .set_precision(2)\
        .set_table_styles(magnify())
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def ConvCount(_df, _ts, _span):
    def myMode(_vals):
        start_idx = 0
        _len = len(_vals)
        #_span = 5
        if _len < _span:
            start_idx = _len
        else:         
            start_idx = _span
        if start_idx > 0:    
            ## ВНИМАНИЕ! При расчете препринятого действия моду следует брать не куммулятивным итогом, а строго в рамках окна визуализации (_span)!!! Поэтому работает ограничение на ряд _vals
            return (int)(stats.mode(_vals[-1*start_idx:])[0]) ##  Как варинат можно применять      return (int)(stats.mode(_vals)[0]) 
        else: 
            return 0
    #conv = _df.groupby(by =['Тема', 'Наименование пакета']).agg({'pred':(sum, len)}).copy()
    #conv = _df.groupby(by =['Тема', 'packUsed']).agg({'pred':(sum, len), 'maxA':(pd.Series.mode)}).copy()
    #conv = _df.groupby(by =['Тема', 'packUsed']).agg({'pred':(sum, len), 'maxA':myMode}).copy()
    conv = _df.groupby(by =['Тема']).agg({'pred':(sum, len), 'A':myMode, 'trueA':myMode}).copy() # Изменил maxA на A!!!!!!
    
    conv[('pred', 'perc')] = conv.apply(lambda x: x[('pred', 'sum')]/x[('pred', 'len')], axis=1)
    
    
    #conv.columns = conv.columns.droplevel() ## !!!drop columns name level, which been added after group by!!!
    conv=conv.reset_index()
    conv.columns=[ 'topic','Response','Sent','bestPack','truePack','Conversion']
    #conv.rename(columns= {\
    #                       'Тема': 'topic'\
    #                       #,'Наименование пакета': 'pack'\
    #                       ,'packUsed': 'pack'\
    #                       ,'sum' : 'Response'\
    #                       ,'len' : 'Sent'\
    #                       ,'perc' : 'Conversion'\
    #                       ,'(maxA,  myMode)':'bestPack'
    #                       ,'(trueA,  myMode)':'truePack'
    #                      }, inplace = True)
    conv['ts'] = _ts
    #conv = conv.set_index(['topic', 'pack', 'ts'])
    conv = conv.set_index(['topic', 'ts'])
    return conv
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def PrepQ(_Q, _S, _A, _ts):
    pdQ = pd.DataFrame(_Q)
    pdQ.index = _S['Тема']
    pd.columns = _A['Наименование пакета']
    pdQ.rename(columns = _A['Наименование пакета'], inplace=True)
    pdQ.reset_index(inplace=True)
    pdQ = pdQ.melt(id_vars=["Тема"], 
            var_name="Наименование пакета", 
            value_name="q") ## transform columns to rows as multiplication with previous rows
    pdQ['ts'] = _ts
    pdQ.rename(columns= {\
                           'Тема': 'topic'\
                           ,'Наименование пакета': 'pack'\
                           ,'q':'Learning rate'
                          }, inplace = True)
    pdQ = pdQ.set_index(['topic', 'pack', 'ts'])
    return pdQ
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

## vsCode, in case of error: "No renderer could be found for mime type "application/vnd.plotly......."
## please set up pio.renderers.default as showed bellow 
## original advise is here:  https://stackoverflow.com/questions/64849484/display-plotly-plot-inside-vs-code
import plotly.io as pio
pio.renderers.default = "notebook_connected"


def DrawLinesOnChart(_dt, _chartTitle):
    fig = px.line(_dt, x=_dt.columns[0], y=_dt.columns[1:])
    fig.update_layout(
        title_text=_chartTitle
    )
    fig.show()
    #display.clear_output(wait=True)
    #display.display(fig)
    #time.sleep(0.1)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def Draw2ScaleLinesOnChart(_dt, _chartTitle):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    _x, _y1, _y2 = _dt.iloc[:,0], _dt.iloc[:,1], _dt.iloc[:,2]
    _xTitle, _y1Title, _y2Title = _dt.columns[0], _dt.columns[1], _dt.columns[2]

    # Add traces
    fig.add_trace(
        go.Scatter(x=_x, y=_y1, name=_y1Title 
                   ,marker=dict(
                        size=8
                        ,cmax=10
                        ,cmin=0
                        ,color=_y1
                        ,colorbar=dict(
                            title="Colorbar"
                        )
                        ,colorscale="Viridis"
                    )
                  ,mode="markers"
                  ) 
       ,secondary_y=False,
    )



    fig.add_trace(
        go.Scatter(x=_x, y=_y2, name=_y2Title),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text=_chartTitle
    )

    # Set x-axis title
    fig.update_xaxes(title_text=_xTitle)

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>"+_y1Title+"</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>"+_y2Title+"</b>", secondary_y=True)

    fig.show()
    display.clear_output(wait=True)
    display.display(fig)
    time.sleep(0.1)
    #//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def DrawMultiCharts(_dt, _chartTitle):
    fig = make_subplots(rows=2, cols=2\
                                                ,specs=[[{}, {}]\
                                                                ,[{"colspan": 2}, None]\
                                                               #, [{}, {}]\
                                                               ]
                                                ,subplot_titles=list(_dt.columns[1:3]) \
                                                                                    +[_dt.columns[4]+' & '+_dt.columns[3]]\
                                                                                    #+list(_dt.columns[5])\
                                              )
    _xTitle, _y1Title, _y2Title = _dt.columns[0], _dt.columns[1], _dt.columns[2]
        
    fig.append_trace(go.Scatter(
        x=_dt.iloc[:,0],
        y=_dt.iloc[:,1],
        name=_dt.columns[1],
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=_dt.iloc[:,0],
        y=_dt.iloc[:,2],
        name=_dt.columns[2],
    ), row=1, col=2)

    fig.append_trace(go.Scatter(
        x=_dt.iloc[:,0],
        y=_dt.iloc[:,3],
        name=_dt.columns[3],
    ), row=2, col=1)
        
    fig.append_trace(go.Scatter(
        x=_dt.iloc[:,0],
        y=_dt.iloc[:,4],
        name=_dt.columns[4],
    ), row=2, col=1)

        #fig.append_trace(go.Scatter(
        #    x=_dt.iloc[:,0],
        #    y=_dt.iloc[:,5],
        #    name=_dt.columns[5],
        #), row=3, col=1)
        # Set x-axis title
    fig.update_xaxes(title_text=_dt.columns[0])

        #fig.update_layout(height=500, width=2000)
    fig.show()

    display.clear_output(wait=True)
    display.display(fig)
    time.sleep(0.01)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# https://www.kaggle.com/jrmistry/plotly-how-to-make-individual-legends-in-subplot - Nice example

def DrawMultiCharts1(_dt, _chartTitle):
    print(_chartTitle)
    fig = make_subplots(rows=2, cols=1\
                                                ,specs=[[{"secondary_y": True}]\
                                                                ,[{}]\
                                                               ]
                                                ,subplot_titles=[_dt.columns[1]+ ' '+_dt.columns[2]+' & '+_dt.columns[3]] \
                                                                                    +[_dt.columns[5]+' & '+_dt.columns[4]]\
                                                ,row_heights=[0.7,0.3]\
                                                ,vertical_spacing=0.1\
                                              )
    fig.append_trace(go.Scatter(
        x=_dt.iloc[:,0],
        y=_dt.iloc[:,1],
        name=_dt.columns[1],
        marker=dict(
                size=8
                ,cmax=10
                ,cmin=0
                ,color='rgb(255,0,0)'
                #,color=_dt.iloc[:,1]
                #,colorbar=dict(
                #    title="PackNum"
                #    ,thickness=10
                #    ,len=0.7
                #    ,x=0
                #    ,y=0.73
                #)
                #,colorscale="Viridis"
            )
            ,mode="markers"
            ,legendgroup = '1'
    ), row=1, col=1)

    #fig.append_trace(go.Scatter(
    #    x=_dt.iloc[:,0],
    #    y=_dt.iloc[:,2],
    #    name=_dt.columns[2],
    #    marker=dict(
    #            size=8
    #            ,cmax=10
    #            ,cmin=0
    #            ,color='rgb(0,0,255)'
    #            #,color=_dt.iloc[:,1]
    #            #,colorbar=dict(
    #            #    title="PackNum"
    #            #    ,thickness=10
    #            #    ,len=0.7
    #            #    ,x=0
    #            #    ,y=0.73
    #            #)
    #            #,colorscale="Viridis"
    #        )
    #        ,mode="markers"
    #        ,legendgroup = '1'
    #), row=1, col=1)

    # Another way to add second curve on a one plot it's also approved
    fig.add_trace(
        go.Scatter(x=_dt.iloc[:,0], y=_dt.iloc[:,3], name=_dt.columns[3], legendgroup = '1'),
        secondary_y=True,
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=_dt.iloc[:,0], y=_dt.iloc[:,2], name=_dt.columns[2],  mode="markers",  legendgroup = '1'),
        secondary_y=False,
        row=1, col=1
    )

    fig.append_trace(go.Scatter(
        x=_dt.iloc[:,0],
        y=_dt.iloc[:,4],
        name=_dt.columns[4],
        legendgroup = '2',
    ), row=2, col=1)
        
    fig.append_trace(go.Scatter(
        x=_dt.iloc[:,0],
        y=_dt.iloc[:,5],
        name=_dt.columns[5],
        legendgroup = '2',
    ), row=2, col=1)

    # Set x-axis title
    fig.update_xaxes(title_text=_dt.columns[0])

    # Set figure size and y-axes titles
    fig.update_layout(height=800, width=1550\
                      ,yaxis1_title= 'Pack'\
                      ,yaxis2_title='Conversion'\
                      ,yaxis3_title='Quantity' 
                      ,legend_tracegroupgap = 400
                      )
    
    fig.show()

    display.clear_output(wait=True)
    display.display(fig)
    time.sleep(0.01)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def DrawDynamicChart(_convHist, _topic, _pack=None):
            convHist1=_convHist.copy()
            convHist1 = convHist1.reset_index()
            if _pack:
                _plotDt = convHist1[(convHist1['topic'] ==_topic) & (convHist1['pack'] == _pack)][['ts','Learning rate','Conversion', 'Response','Sent','bestPack','truePack']].copy()
            else:
                _plotDt = convHist1[(convHist1['topic'] ==_topic)][['ts','Learning rate','Conversion', 'Response','Sent','bestPack','truePack']].copy()
                
            #DrawMultiCharts(_plotDt, "Conversion & Learning rate")
            DrawMultiCharts1(_plotDt.iloc[:, [0,5,6,2,4,3]], _topic)
            #Draw2ScaleLinesOnChart(_plotDt, 'Conversion & Learning rate') 
            #_plotDt = convHist1[(convHist1['topic'] ==_topic) & (convHist1['pack'] == _pack)][['ts','Response','Sent']].copy()
            #DrawLinesOnChart(_plotDt, 'Sent & Response')
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def CommonChart(_complLeadsShort, _topic, _pack=None):
    def myMode(_vals):
        return (int)(pd.Series.mode(_vals)[0])
    _complLeadsShort['ts'] = _complLeadsShort.index#//100
    _df = _complLeadsShort[(_complLeadsShort['Тема'] == _topic)  & (~_complLeadsShort['maxA'].isna())].reset_index().copy()
    conv = _df.groupby(by =['Тема', 'ts']).agg({'A':myMode, 'trueA':myMode, 'pred':(sum, len)}).copy()
    conv[[('pred', 'sum'), ('pred', 'len')]] = conv[[('pred', 'sum'), ('pred', 'len')]].cumsum()
    conv[('pred', 'perc')] = conv.apply(lambda x: x[('pred', 'sum')]/x[('pred', 'len')], axis=1)
    #conv.columns = conv.columns.droplevel() ## !!!drop columns name level, which been added after group by!!!
    conv=conv.reset_index()
    #conv.rename(columns= {\
    #                       'Тема': 'topic'\
    #                       #,'Наименование пакета': 'pack'\
    #                       ,'sum' : 'Response'\
    #                       ,'len' : 'Sent'\
    #                       ,'perc' : 'Conversion'\
    #                       ,'myMode':'bestPack'
    #                      }, inplace = True)
    conv.columns=[ 'topic','ts','bestPack','truePack','Response','Sent','Conversion']
    conv = conv.set_index(['topic',  'ts'])
    #DrawMultiCharts1(conv.reset_index().iloc[:, [1,2,5,4,3]], _topic)
    DrawMultiCharts1(conv.reset_index().iloc[:, [1,2,3,6,5,4]], _topic)
    print(_topic,'\n',_pack)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////