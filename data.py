#%%
# loading neceassery libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df=pd.read_csv('hfcrd.csv')
df1=df.head()
# %%
df.info()
df.columns
#%%
# cambiar nombres de las columnas en español
column_translation = {
    'age': 'edad',
    'anaemia': 'anemia',
    'creatinine_phosphokinase': 'creatinina_fosfoquinasa',
    'diabetes': 'diabetes',
    'ejection_fraction': 'fraccion_eyeccion',
    'high_blood_pressure': 'presion_arterial_alta',
    'platelets': 'plaquetas',
    'serum_creatinine': 'creatinina_serica',
    'serum_sodium': 'sodio_serico',
    'sex': 'sexo',
    'smoking': 'fumador',
    'time': 'tiempo',
    'DEATH_EVENT': 'EVENTO_MUERTE'
}
df.rename(columns=column_translation, inplace=True)
#%%
df1=df.head()
# %%
# html_table = df1.to_html(index=False)
# with open('table.html', 'w') as f:
#     f.write(html_table)
# %%
