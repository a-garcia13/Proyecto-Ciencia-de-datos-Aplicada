import streamlit as st
# Manejo de datos
import pandas as pd
import numpy as np

# Visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns

# Entrenamiento del modelo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz

# Librerías extras
from subprocess import call
import itertools


def main():
    institutions, X_train, y_train, X_test, y_test = preprocess()
    random = model(X_train, y_train)
    st.title('Sistema de proyeción de postulaciones semestrales para negociación de cupos')
    option = st.sidebar.selectbox('Options:',
                                  ['Proyeción por país', 'Proyeción por institución', 'Institución nueva'])
    if option == 'Proyeción por país':
        st.subheader("Proyección por país:")
        with st.form("form_1"):
            col1, col2 = st.columns([3, 3])
            with col1:
                country = st.selectbox('País:', institutions["Country"].sort_values().unique())
            with col2:
                semestre_1 = st.selectbox("Semestre:", ['Primer Semestre 2024', 'Segundo Semestre 2024'])
            submit_button_1 = st.form_submit_button(label='Estimar')
            if submit_button_1:
                first_semester = 0
                second_semester = 1
                if semestre_1 == 'Primer Semestre 2024':
                    first_semester = 1
                    second_semester = 0
                contain_values = institutions[institutions['Country'] == country]
                contain_values['Sem_First Semester'] = first_semester
                contain_values['Sem_Second Semester'] = second_semester
                ready = process(contain_values)
                result = pd.DataFrame(random.predict(ready), columns=['Numero de postulaciones estimado'])
                df1 = contain_values[['Country', 'Institution']].reset_index().drop('index',  axis=1)
                df1 = pd.concat([df1, result], axis=1)
                df1 = df1.drop_duplicates()
                st.write(df1)

    if option == 'Proyeción por institución':
        st.subheader("Proyección por institución:")
        with st.form("form_2"):
            col3, col4 = st.columns([3, 3])
            with col3:
                institution = st.selectbox('Institución:', institutions["Institution"].sort_values().unique())
            with col4:
                semestre_2 = st.selectbox("Semestre:", ['Primer Semestre 2024', 'Segundo Semestre 2024'])
            submit_button_2 = st.form_submit_button(label='Estimar')
            if submit_button_2:
                first_semester = 0
                second_semester = 1
                if semestre_2 == 'Primer Semestre 2024':
                    first_semester = 1
                    second_semester = 0
                contain_values = institutions[institutions['Institution'] == institution]
                contain_values['Sem_First Semester'] = first_semester
                contain_values['Sem_Second Semester'] = second_semester
                ready = process(contain_values)
                result = pd.DataFrame(random.predict(ready), columns=['Numero de postulaciones estimado'])
                df1 = contain_values[['Country', 'Institution']].reset_index().drop('index',  axis=1)
                df1 = pd.concat([df1, result], axis=1)
                df1 = df1.drop_duplicates()
                st.write(df1)


@st.experimental_singleton
def preprocess():
    url_stay_wishes = 'https://github.com/a-garcia13/Proyecto-Ciencia-de-datos-Aplicada/blob/main/Data/AcademicMoveWishesOutgoing%20(Sat%20Sep%2017%202022).xlsx?raw=true'
    stay_wishes = pd.read_excel(url_stay_wishes)
    stay_wishes = stay_wishes[stay_wishes['Level'] == 'Undergraduate / Bachelor']
    stay_wishes = stay_wishes.drop(['Level'], axis=1)

    url_institutions = 'https://github.com/a-garcia13/Proyecto-Ciencia-de-datos-Aplicada/blob/main/Data/Institutions%20(Sat%20Sep%2017%202022).xlsx?raw=true'
    institutions = pd.read_excel(url_institutions)

    url_relations = 'https://github.com/a-garcia13/Proyecto-Ciencia-de-datos-Aplicada/blob/main/Data/Relations%20(Sat%20Sep%2017%202022).xlsx?raw=true'
    relations = pd.read_excel(url_relations)

    data = stay_wishes.copy()
    data = data[data['Form'].str.contains('- Outgoing')]
    frm = relations[['Relation ID', 'Frameworks', 'Country', 'Main External Institutions']]
    data['Relation ID'] = data['Relation: ID']
    data.drop('Relation: ID', axis=1)
    data = pd.merge(data, frm, how='left', on='Relation ID')
    data.loc[data['Frameworks_x'].isna(), 'Frameworks_x'] = data['Frameworks_y']
    data = data[data['Frameworks_x'].notna()]
    data['Frameworks'] = data['Frameworks_x']
    data = data.drop(['Frameworks_y', 'Frameworks_x'], axis=1)
    data.loc[data["Frameworks"] == "Exchange Student Undergraduate,SMILE", "Frameworks"] = 'SMILE'
    data.loc[data["Frameworks"] == "Exchange Student Undergraduate,Study Abroad", "Frameworks"] = 'Study Abroad'
    data.loc[data["Frameworks"] == "CINDA,Exchange Student Undergraduate", "Frameworks"] = 'CINDA'
    data.loc[data["Frameworks"] == "Exchange Student Undergraduate, SMILE", "Frameworks"] = 'SMILE'
    data = data[data['Frameworks'] != 'SURF']
    data = data[data['Frameworks'] != 'Research Internship']
    data = data[data['Frameworks'] != 'HUC']
    data = data[data['Frameworks'] != 'Virtual Mobility Program']
    data = data[data['Frameworks'] != 'Medical Clerkship']
    data = data[data['Frameworks'] != 'Proyecto de Grado']
    data = data[data['Frameworks'] != 'Sígueme']
    data = data[data['Frameworks'] != 'GE3']
    data = data[data['Frameworks'] != 'Sui Iuris']
    data = data[data['Relation ID'].notna()]
    data.loc[data['Country_x'].isna(), 'Country_x'] = data['Country_y']
    data.loc[data['Institution'].isna(), 'Institution'] = data['Main External Institutions']
    data.loc[data['Institution'] == 'Universidad de Los Andes', 'Country_x'] = data['Country_y']
    data.loc[data['Institution'] == 'Universidad de Los Andes', 'Institution'] = data['Main External Institutions']
    data = data[data['Relation: Direction'] != 'Incoming']
    data['Country'] = data['Country_x']
    data = data.drop(['Stay wish: ID', 'Person: ID', 'Form', 'Rank',
                      'Status selection', 'Status offer', 'Sub institution', 'Country_x',
                      'Duration periods', 'Stay: ID', 'Relation: ID', 'Frameworks',
                      'Person: Sub institution', 'Relation: Direction', 'Relation: Level',
                      'Stay: Status', 'Stay: Degree programme', 'Relation: Reach', 'Country_y',
                      'Main External Institutions'], axis=1)
    data = data.groupby(['Country', 'Institution', 'Start period']).size().reset_index(name='Count')
    data['Start period'] = data['Start period'].str.replace('\d+', '')
    data['Start period'] = data['Start period'].str.strip()
    data = data.groupby(['Country', 'Institution', 'Start period'])['Count'].mean().reset_index(name='Mean')

    inst = institutions.copy()
    inst = inst.drop(['Institution type', 'Country', 'Language cerf score 1',
                      'Language cerf score 2', 'Language requirement 3', 'Language cerf score 3'], axis=1)
    inst.loc[inst['Minimum GPA/4'] == 'C2', 'Minimum GPA/4'] = 2.6
    inst['Minimum GPA/4'] = inst['Minimum GPA/4'].str.replace(',', '.')
    inst['Minimum GPA/4'] = inst['Minimum GPA/4'].astype(float)
    inst['Language requirement 1'] = inst['Language requirement 1'].astype(str)
    inst['Language requirement 2'] = inst['Language requirement 2'].astype(str)
    inst['Official Language'] = inst['Official Language'].str.replace(", ", ",")
    inst["Languages"] = inst[['Official Language', 'Language requirement 1', 'Language requirement 2']].apply(','.join,
                                                                                                              axis=1)
    inst = inst.drop(['Official Language', 'Language requirement 1', 'Language requirement 2'], axis=1)
    inst['Languages'] = inst['Languages'].str.split(',')
    data = pd.merge(data, inst, how='left', left_on='Institution', right_on='Name')
    data = data.drop(['Name'], axis=1)
    data.loc[data['Region 2'].isna(), 'Region 2'] = 0
    data.loc[data['Region 2'] == 'Latin America and the Caribbean', 'Region 2'] = 1
    data.rename(columns={'Region 2': 'Latin America and the Caribbean'}, inplace=True)

    label = data['Languages']
    mlb = MultiLabelBinarizer()
    one_hot = pd.DataFrame(data=mlb.fit_transform(label), columns=mlb.classes_)
    data.reset_index(drop=True, inplace=True)
    one_hot.reset_index(drop=True, inplace=True)
    data = pd.concat([data, one_hot], axis=1)
    data = data.drop(['nan', 'Languages'], axis=1)

    rel = relations.copy()
    stay_op = rel.loc[rel['Relation type'] == 'Stay opportunity']
    stay_op = stay_op.loc[rel['Direction'] == 'Outgoing']
    stay_op = stay_op.loc[rel['Level'].str.contains('Undergraduate / Bachelor', na=False)]
    stay_op = stay_op[stay_op['Degree programme'].notna()]
    stay_op['Degree programme'] = stay_op['Degree programme'].astype(str)
    stay_op['Degree programme'] = stay_op['Degree programme'].str.replace("|", ",")
    stay_op['Degree programme'] = stay_op['Degree programme'].str.replace(", ", "")
    stay_op['Degree programme'] = stay_op['Degree programme'].str.split(',')
    label = stay_op['Degree programme']
    mlb = MultiLabelBinarizer()
    one_hot = pd.DataFrame(data=mlb.fit_transform(label), columns=mlb.classes_)
    stay_op.reset_index(drop=True, inplace=True)
    one_hot.reset_index(drop=True, inplace=True)
    one_hot = one_hot.loc[:, one_hot.columns.str.contains("Bsc")]
    stay_op = pd.concat([stay_op, one_hot], axis=1)
    stay_op = stay_op.drop(['Degree programme'], axis=1)
    stay_op_count = stay_op.groupby(['Country', 'Main External Institutions'])['Administration Bsc',
                                                                               'Anthropology Bsc', 'Architecture Bsc', 'Art History Bsc', 'Arts Bsc',
                                                                               'Biology Bsc', 'Biomedical Engineering Bsc', 'Chemical Engineering Bsc',
                                                                               'Chemistry Bsc', 'Civil Engineering Bsc', 'Design Bsc',
                                                                               'Digital Narrative Bsc', 'Economics Bsc', 'Education Bsc / Arts',
                                                                               'Education Bsc / Biology', 'Education Bsc / Early Childhood',
                                                                               'Education Bsc / History', 'Education Bsc / Humanities',
                                                                               'Education Bsc / Mathematics', 'Education Bsc / Philosophy',
                                                                               'Education Bsc / Spanish and Philology', 'Electrical Engineering Bsc',
                                                                               'Electronic Engineering Bsc', 'Environmental Engineering Bsc',
                                                                               'Geosciences Bsc', 'Global Studies Bsc',
                                                                               'Government and Public Policy Bsc', 'History Bsc',
                                                                               'Industrial Engineering Bsc', 'International Accounting Bsc',
                                                                               'Languages and Culture Bsc', 'Law Bsc', 'Literature Bsc',
                                                                               'Mathematics Bsc', 'Mechanical Engineering Bsc', 'Microbiology Bsc',
                                                                               'Music Bsc', 'Other Bsc', 'Philosophy Bsc', 'Physics Bsc',
                                                                               'Political Science Bsc', 'Psychology Bsc',
                                                                               'Systems and Computing Engineering Bsc'].agg(
        'sum').reset_index()
    stay_op = stay_op.groupby(['Country', 'Main External Institutions']).size().reset_index(name='SO Count')
    stay_op = pd.merge(stay_op, stay_op_count, how='left', on=['Country', 'Main External Institutions'])
    stay_op = stay_op.drop('Country', axis=1)
    partnerships = rel.loc[rel['Relation type'] == 'Partnership']
    partnerships = partnerships[partnerships['Reach'].notna()]
    y = pd.get_dummies(partnerships.Reach, prefix='Reach')
    partnerships = partnerships.join(y).drop(columns=['Reach'])
    partnerships = partnerships.groupby(['Country', 'Main External Institutions'])[
        'Reach_Specific agreement', 'Reach_University wide'].agg('sum').reset_index()
    partnerships = partnerships.drop('Country', axis=1)
    data = pd.merge(data, partnerships, how='left', left_on='Institution', right_on='Main External Institutions')
    data = data.drop('Main External Institutions', axis=1)
    data = pd.merge(data, stay_op, how='left', left_on='Institution', right_on='Main External Institutions')
    data = data.drop('Main External Institutions', axis=1)
    data = data.fillna(0)
    y = pd.get_dummies(data['Country'], prefix='Country')
    z = pd.get_dummies(data['Start period'], prefix='Sem')
    f = pd.get_dummies(data['Region 1'], prefix='Region')
    x = pd.get_dummies(data['Continent'], prefix='Continent')
    data = data.join(y)
    data = data.join(z)
    data = data.join(f)
    data = data.join(x)
    institutions = data.copy()
    data = data.drop(['Continent', 'Region 1', 'Start period', 'Country', 'Institution', 'Institution: ID', 'City', 'Other Bsc'], axis=1)
    train, test = train_test_split(data, test_size=0.2, random_state=33)
    X_train, y_train = train.drop(['Mean'], axis=1), train['Mean']
    X_test, y_test = test.drop(['Mean'], axis=1), test['Mean']

    return institutions, X_train, y_train, X_test, y_test

@st.experimental_singleton
def process(data):
    data = data.drop(['Continent', 'Region 1', 'Start period', 'Country', 'Institution', 'Institution: ID', 'City', 'Other Bsc', 'Mean'], axis=1)
    return data

@st.experimental_singleton
def model(X_train, y_train):
    estimators = [
        ("normalizar", StandardScaler()),
        ("rfc", RandomForestRegressor()),
    ]

    pipe = Pipeline(estimators)

    parameters = {
        'normalizar': [StandardScaler()],
        'rfc__n_estimators': [50],
        'rfc__max_depth': [10],
        'rfc__min_samples_leaf': [2]
    }
    grid_search = GridSearchCV(pipe, parameters, verbose=2, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


if __name__ == '__main__':
    main()
