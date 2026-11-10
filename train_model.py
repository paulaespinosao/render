from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

RNG = np.random.default_rng(42)
N = 1800
programas = np.array(['Ingeniería', 'Negocios', 'Salud', 'Diseño'])
df = pd.DataFrame({
    'student_id': [f'E{i:05d}' for i in range(1, N + 1)],
    'programa': RNG.choice(programas, N, p=[.32, .28, .25, .15]),
    'semestre': RNG.integers(1, 9, N),
    'promedio': np.clip(RNG.normal(77, 11, N), 35, 100).round(1),
    'asistencia_pct': np.clip(RNG.normal(82, 13, N), 20, 100).round(1),
    'entregas_tarde': np.clip(RNG.poisson(2.2, N), 0, 12),
    'horas_plataforma': np.clip(RNG.gamma(3.2, 2.2, N), .2, 30).round(1),
    'apoyo_financiero': RNG.choice(['Sí', 'No'], N, p=[.38, .62]),
    'satisfaccion': RNG.integers(1, 6, N),
})
logit = (-5.0 + .055*(70-df.promedio) + .045*(75-df.asistencia_pct)
         + .24*df.entregas_tarde - .09*df.horas_plataforma
         + .42*(df.satisfaccion <= 2) + .28*(df.semestre == 1)
         + .18*(df.apoyo_financiero == 'No'))
prob = 1/(1+np.exp(-logit))
df['abandono'] = RNG.binomial(1, np.clip(prob, .01, .92))

features = ['programa','semestre','promedio','asistencia_pct','entregas_tarde','horas_plataforma','apoyo_financiero','satisfaccion']
cat = ['programa','apoyo_financiero']; num = [c for c in features if c not in cat]
pre = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),('scale', StandardScaler())]), num),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat)
])
model = Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=1200, class_weight='balanced'))])
X_train, X_test, y_train, y_test = train_test_split(df[features], df['abandono'], test_size=.25, stratify=df['abandono'], random_state=42)
model.fit(X_train, y_train)
p = model.predict_proba(X_test)[:,1]
print(f'ROC-AUC: {roc_auc_score(y_test,p):.3f}')
print(classification_report(y_test, p >= .5, digits=3))
Path('data').mkdir(exist_ok=True); Path('model').mkdir(exist_ok=True)
df['riesgo_prob'] = model.predict_proba(df[features])[:,1]
df.to_csv('data/estudiantes.csv', index=False)
joblib.dump({'model':model,'features':features}, 'model/modelo_abandono.joblib')
print('Archivos creados en data/ y model/.')
