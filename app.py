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
    countries, courses, languages, institutions, X_train, y_train, X_test, y_test = preprocess()
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
                semestre_3 = st.selectbox("Semestre:", ['Primer Semestre 2024', 'Segundo Semestre 2024'])
            submit_button_3 = st.form_submit_button(label='Estimar')
            if submit_button_3:
                first_semester = 0
                second_semester = 1
                if semestre_3 == 'Primer Semestre 2024':
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

    if option == 'Institución nueva':
        # se necesita: pais, gpa minimo, idioma(s), numero de convenios especificos y generales, numero de so, programas, semestre.
        st.subheader("Institución nueva:")
        with st.form("form_3"):
            col5, col6 = st.columns([3, 3])
            with col5:
                universidad = st.text_input('Nombre de la universidad')
                country_2 = st.selectbox('País:', countries["Country"].sort_values().unique())
                promedio = st.number_input("Promedio Minimo sobre 4:", 2.6, 4.0, step=0.1, format="%.2f")
                generales = st.number_input("Numero de convenios generales", 0.0, 5.0, step=1.0, format="%.2f")
                especificos = st.number_input("Numero de convenios especificos", 0.0, 12.0, step=1.0, format="%.2f")
            with col6:
                semestre_2 = st.selectbox("Semestre:", ['Primer Semestre 2024', 'Segundo Semestre 2024'])
                languajes = st.multiselect("Idiomas disponibles para intercambios:",
                                          languages['Language'].sort_values().unique())
                stay_opportunites = st.number_input("Numero de Stay Opportunities", 0.0, 20.0, step=1.0, format="%.2f")
                program = st.multiselect('Programas con convenios especificos:',
                                       courses["Name"].sort_values().unique())
            submit_button_2 = st.form_submit_button(label='Estimar')
            if submit_button_2:
                if languajes is None:
                    languajes = []
                    st.write('lista de idiomas vacia')
                first_semester = 0
                second_semester = 1
                if semestre_2 == 'Primer Semestre 2024':
                    first_semester = 1
                    second_semester = 0
                contain_values = countries[countries['Country'] == country_2]
                contain_values = contain_values[['Region 1', 'Region 2', 'Continent', 'Official Language', 'Country']].reset_index().drop('index',  axis=1)
                contain_values = contain_values.drop_duplicates().reset_index().drop('index',  axis=1)
                contain_values = new_institution(contain_values, promedio, generales, especificos, first_semester, second_semester, languajes, program, stay_opportunites)
                result = random.predict(contain_values)
                uni = pd.DataFrame(columns=['Country','Institution','Numero de postulaciones estimado'])
                uni.loc[0] = [country_2, universidad, result]
                st.write(uni)


@st.experimental_singleton
def preprocess():
    url_stay_wishes = 'https://github.com/a-garcia13/Proyecto-Ciencia-de-datos-Aplicada/blob/main/Data/AcademicMoveWishesOutgoing%20(Sat%20Sep%2017%202022).xlsx?raw=true'
    stay_wishes = pd.read_excel(url_stay_wishes)
    stay_wishes = stay_wishes[stay_wishes['Level'] == 'Undergraduate / Bachelor']
    stay_wishes = stay_wishes.drop(['Level'], axis=1)

    url_institutions = 'https://github.com/a-garcia13/Proyecto-Ciencia-de-datos-Aplicada/blob/main/Data/Institutions%20(Sat%20Sep%2017%202022).xlsx?raw=true'
    institutions = pd.read_excel(url_institutions)
    countries = institutions.copy()

    url_courses = 'https://github.com/a-garcia13/Proyecto-Ciencia-de-datos-Aplicada/blob/main/Data/Courses%20(Wed%20Nov%2002%202022).xlsx?raw=true'
    # Datos de los pregrados con sus respectivas facultades y departamentos
    courses = pd.read_excel(url_courses)

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
    languages = pd.DataFrame(mlb.classes_, columns=['Language'])
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

    return countries, courses, languages, institutions, X_train, y_train, X_test, y_test

@st.experimental_singleton
def process(data):
    data = data.drop(['Continent', 'Region 1', 'Start period', 'Country', 'Institution', 'Institution: ID', 'City', 'Other Bsc', 'Mean'], axis=1)
    return data

@st.experimental_singleton
def new_institution(pais, promedio, generales, especificos, semestre_1, semestre_2, lenguajes, programas, stays):

    latin = 0
    if pais['Region 2'].iloc[0] == 'Latin America and the Caribbean':
        latin = 1
    pais = pais.drop('Region 2', axis=1)
    # 'Region 1', 'Region 2', 'Continent', 'Official Language', 'Country'
    pais['Latin America and the Caribbean'] = latin

    pais['Minimum GPA/4'] = promedio

    language_list = ['Aymara language', "Bokmål", 'Danish', 'Dutch', 'English', 'Finnish', 'French', 'German',
                     'Indonesian', 'Italian', 'Japanese', 'Korean', 'Lule', 'Māori language', 'Norwegian', 'Nynorsk',
                     'Polish', 'Portuguese', 'Quechuan languages', 'Romansh', 'Russian', 'Saami', 'Slovene language',
                     'Spanish', 'Standard Chinese', 'Swedish']

    values_language = []

    for lang in language_list:
        values_language.append(0)

    official_lang = pais['Official Language'].iloc[0]

    pais = pais.drop('Official Language', axis=1)

    pos = 0
    for lenguaj in language_list:
        for lang in lenguajes:
            if lang == lenguaj:
                values_language[pos] = 1
        if lenguaj == official_lang:
            values_language[pos] = 1
        pos = pos + 1

    temp_df = pd.DataFrame(columns=language_list)
    temp_df.loc[0] = values_language
    pais = pd.concat([pais, temp_df], axis=1)

    pais['Reach_Specific agreement'] = especificos
    pais['Reach_University wide'] = generales
    pais['SO Count'] = stays

    degree_list = ['Administration Bsc', 'Anthropology Bsc', 'Architecture Bsc', 'Art History Bsc', 'Arts Bsc',
                   'Biology Bsc',
                   'Biomedical Engineering Bsc', 'Chemical Engineering Bsc', 'Chemistry Bsc', 'Civil Engineering Bsc',
                   'Design Bsc',
                   'Digital Narrative Bsc', 'Economics Bsc', 'Education Bsc / Arts', 'Education Bsc / Biology',
                   'Education Bsc / Early Childhood',
                   'Education Bsc / History', 'Education Bsc / Humanities', 'Education Bsc / Mathematics',
                   'Education Bsc / Philosophy',
                   'Education Bsc / Spanish and Philology', 'Electrical Engineering Bsc', 'Electronic Engineering Bsc',
                   'Environmental Engineering Bsc', 'Geosciences Bsc', 'Global Studies Bsc',
                   'Government and Public Policy Bsc',
                   'History Bsc', 'Industrial Engineering Bsc', 'International Accounting Bsc',
                   'Languages and Culture Bsc',
                   'Law Bsc', 'Literature Bsc', 'Mathematics Bsc', 'Mechanical Engineering Bsc', 'Microbiology Bsc',
                   'Music Bsc',
                   'Philosophy Bsc', 'Physics Bsc', 'Political Science Bsc', 'Psychology Bsc',
                   'Systems and Computing Engineering Bsc']

    values_degree = []

    for degree in degree_list:
        if generales > 0:
            values_degree.append(1)
        else:
            values_degree.append(0)

    for program in programas:
        pos = 0
        for degree in degree_list:
            if program == degree:
                values_degree[pos] = values_degree[pos] + 1
            pos = pos + 1

    temp_df = pd.DataFrame(columns=degree_list)
    temp_df.loc[0] = values_degree
    pais = pd.concat([pais, temp_df], axis=1)

    country_list = ['Country_Argentina',
                    'Country_Australia', 'Country_Belgium', 'Country_Brazil', 'Country_Canada', 'Country_Chile',
                    'Country_China', 'Country_Colombia', 'Country_Costa Rica', 'Country_Denmark',
                    'Country_Dominican Republic',
                    'Country_Finland', 'Country_France', 'Country_Germany', 'Country_Indonesia', 'Country_Italy',
                    'Country_Japan', 'Country_Korea, Republic of', 'Country_Mexico', 'Country_Netherlands',
                    'Country_New Zealand',
                    'Country_Norway', 'Country_Peru', 'Country_Poland', 'Country_Portugal',
                    'Country_Russian Federation', 'Country_Slovenia', 'Country_Spain', 'Country_Sweden', 'Country_Switzerland', 'Country_Taiwan',
                    'Country_United Kingdom', 'Country_United States', 'Country_Uruguay']

    values_country = []

    for country in country_list:
        if pais['Country'].iloc[0] in country:
            values_country.append(1)
        else:
            values_country.append(0)

    temp_df = pd.DataFrame(columns=country_list)
    temp_df.loc[0] = values_country
    pais = pd.concat([pais, temp_df], axis=1)
    pais = pais.drop('Country', axis=1)

    pais['Sem_First Semester'] = semestre_1
    pais['Sem_Second Semester'] = semestre_2

    region_list = ['Region_Australia and New Zealand', 'Region_Caribbean', 'Region_Central America',
                    'Region_Eastern Asia', 'Region_Eastern Europe', 'Region_Northern America', 'Region_Northern Europe',
                    'Region_South America', 'Region_South-eastern Asia', 'Region_Southern Europe', 'Region_Western Europe']
    values_region = []

    for region in region_list:
        if pais['Region 1'].iloc[0] in region:
            values_region.append(1)
        else:
            values_region.append(0)

    temp_df = pd.DataFrame(columns=region_list)
    temp_df.loc[0] = values_region
    pais = pd.concat([pais, temp_df], axis=1)
    pais = pais.drop('Region 1', axis=1)

    continent_list = ['Continent_Asia', 'Continent_Europe', 'Continent_North America',	'Continent_Oceania',
                      'Continent_South America']
    values_continent = []

    for continent in continent_list:
        if pais['Continent'].iloc[0] in continent:
            values_continent.append(1)
        else:
            values_continent.append(0)

    temp_df = pd.DataFrame(columns=continent_list)
    temp_df.loc[0] = values_continent
    pais = pd.concat([pais, temp_df], axis=1)
    pais = pais.drop('Continent', axis=1)

    return pais

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
