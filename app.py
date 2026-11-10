import os
from pathlib import Path
import joblib
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table, no_update
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go

DATA = Path('data/estudiantes.csv'); MODEL = Path('model/modelo_abandono.joblib')
if not DATA.exists() or not MODEL.exists():
    raise FileNotFoundError('Ejecuta primero: python train_model.py')
df = pd.read_csv(DATA)
bundle = joblib.load(MODEL); model=bundle['model']; FEATURES=bundle['features']

def risk_label(p): return 'Alto' if p >= .65 else ('Medio' if p >= .35 else 'Bajo')
df['nivel_riesgo']=df['riesgo_prob'].map(risk_label)
COLORS={'bg':'#091426','panel':'#101f3d','cyan':'#22d3ee','violet':'#8b5cf6','text':'#e6eefc','muted':'#9fb0ca'}
app=Dash(__name__, title='Alertas académicas IA'); server=app.server
card=lambda title,id_: html.Div([html.Small(title),html.H2(id=id_)],className='card')
app.layout=html.Div([
 html.Div([html.H1('Centro de alertas académicas'),html.P('Exploración e inferencia responsable con datos sintéticos')],className='hero'),
 html.Div([dcc.Dropdown(sorted(df.programa.unique()),multi=True,id='programa',placeholder='Programa'),dcc.Dropdown(sorted(df.semestre.unique()),multi=True,id='semestre',placeholder='Semestre'),dcc.Dropdown(['Bajo','Medio','Alto'],multi=True,id='riesgo',placeholder='Nivel de riesgo')],className='filters'),
 html.Div([card('Estudiantes','kpi-n'),card('Tasa observada','kpi-rate'),card('Promedio','kpi-avg'),card('Asistencia','kpi-att')],className='kpis'),
 html.Div([dcc.Graph(id='scatter'),dcc.Graph(id='hist')],className='grid2'),
 html.Div([html.H3('Casos priorizados'),dash_table.DataTable(id='table',page_size=8,sort_action='native',style_table={'overflowX':'auto'},style_cell={'backgroundColor':COLORS['panel'],'color':COLORS['text'],'border':'1px solid #294165','padding':'8px'},style_header={'backgroundColor':'#17305b','fontWeight':'bold'})],className='panel'),
 html.Div([html.H2('Simular estudiante'),html.Div([
   dcc.Dropdown(sorted(df.programa.unique()),'Ingeniería',id='s-programa'),dcc.Input(id='s-semestre',type='number',min=1,max=8,value=1),dcc.Input(id='s-promedio',type='number',min=0,max=100,value=75),dcc.Input(id='s-asistencia',type='number',min=0,max=100,value=80),dcc.Input(id='s-tarde',type='number',min=0,max=20,value=2),dcc.Input(id='s-horas',type='number',min=0,max=60,value=6),dcc.Dropdown(['Sí','No'],'No',id='s-apoyo'),dcc.Slider(1,5,1,value=3,id='s-sat'),html.Button('Estimar riesgo',id='predict',n_clicks=0)
 ],className='formgrid'),html.Div(id='prediction',className='prediction',role='status'),html.P('Uso responsable: la alerta apoya una revisión humana y no debe automatizar sanciones o exclusión.',className='notice')],className='panel')
],className='shell')

@callback(Output('kpi-n','children'),Output('kpi-rate','children'),Output('kpi-avg','children'),Output('kpi-att','children'),Output('scatter','figure'),Output('hist','figure'),Output('table','data'),Output('table','columns'),Input('programa','value'),Input('semestre','value'),Input('riesgo','value'))
def update_dashboard(programas,semestres,riesgos):
    x=df.copy()
    if programas: x=x[x.programa.isin(programas)]
    if semestres: x=x[x.semestre.isin(semestres)]
    if riesgos: x=x[x.nivel_riesgo.isin(riesgos)]
    if x.empty:
        empty=go.Figure().add_annotation(text='Sin resultados para los filtros seleccionados',showarrow=False)
        empty.update_layout(template='plotly_dark',paper_bgcolor=COLORS['panel'],plot_bgcolor=COLORS['panel'])
        return '0','0%','-','-',empty,empty,[],[]
    sc=px.scatter(x,x='asistencia_pct',y='promedio',color='nivel_riesgo',hover_data=['student_id','programa','riesgo_prob'],color_discrete_map={'Bajo':'#22d3ee','Medio':'#f59e0b','Alto':'#ef4444'},title='Promedio vs. asistencia')
    hi=px.histogram(x,x='riesgo_prob',nbins=20,color='nivel_riesgo',title='Distribución de probabilidad',color_discrete_map={'Bajo':'#22d3ee','Medio':'#f59e0b','Alto':'#ef4444'})
    for fig in (sc,hi): fig.update_layout(template='plotly_dark',paper_bgcolor=COLORS['panel'],plot_bgcolor=COLORS['panel'],legend_title_text='Riesgo')
    cols=['student_id','programa','semestre','promedio','asistencia_pct','riesgo_prob','nivel_riesgo']
    tab=x.nlargest(30,'riesgo_prob')[cols].copy(); tab['riesgo_prob']=(tab['riesgo_prob']*100).round(1).astype(str)+'%'
    return f'{len(x):,}',f'{100*x.abandono.mean():.1f}%',f'{x.promedio.mean():.1f}',f'{x.asistencia_pct.mean():.1f}%',sc,hi,tab.to_dict('records'),[{'name':c.replace('_',' ').title(),'id':c} for c in cols]

@callback(Output('prediction','children'),Input('predict','n_clicks'),State('s-programa','value'),State('s-semestre','value'),State('s-promedio','value'),State('s-asistencia','value'),State('s-tarde','value'),State('s-horas','value'),State('s-apoyo','value'),State('s-sat','value'),prevent_initial_call=True)
def predict(n,programa,semestre,promedio,asistencia,tarde,horas,apoyo,sat):
    vals=[programa,semestre,promedio,asistencia,tarde,horas,apoyo,sat]
    if any(v is None for v in vals): return 'Completa todos los campos.'
    row=pd.DataFrame([{k:v for k,v in zip(FEATURES,vals)}])
    p=float(model.predict_proba(row)[0,1]); level=risk_label(p)
    suggestions=[]
    if asistencia<75: suggestions.append('revisar barreras de asistencia')
    if promedio<70: suggestions.append('ofrecer tutoría académica')
    if horas<4: suggestions.append('explorar participación en plataforma')
    action=', '.join(suggestions) if suggestions else 'mantener seguimiento preventivo'
    return html.Div([html.H3(f'Riesgo {level}: {p:.1%}'),html.P(f'Acción sugerida: {action}.')])

if __name__=='__main__':
    app.run(debug=os.getenv('DASH_DEBUG','false').lower()=='true',host='0.0.0.0',port=int(os.getenv('PORT','8050')))
