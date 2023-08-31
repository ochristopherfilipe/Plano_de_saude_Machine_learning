# Importando as bibliotecas
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import plotly.graph_objects as go

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(layout='wide')

st.title('Previsão de Gastos com Plano de Saúde')
st.subheader('Nossa máquina de predição de custos de plano de saúde é baseada em um conjunto de dados anônimo de um dos maiores hospitais do Brasil. Exploramos informações relevantes, como idade, índice de massa corporal (IMC), número de filhos, região geográfica, sexo e tabagismo, para entender como os gastos com planos de saúde são calculados. No final da análise, oferecemos um ambiente de deploy interativo onde você pode inserir suas informações e descobrir uma estimativa do custo anual do plano de saúde com base em nosso modelo preditivo. Nosso objetivo é fornecer insights valiosos sobre como fatores individuais afetam os custos de saúde, ajudando nas decisões de cobertura médica de forma confidencial e informada.')


# Carregando a base de dados
train_data = pd.read_csv('Train_Data.csv')


@st.cache_data
def load_data(nrows):
    data = train_data
    return data
with st.container():
    data_load_state = st.text('Carregando o dataset...')
    data = load_data(10000)
    data_load_state.text("")

    if st.checkbox('Mostrar RAW do dataset'):
        st.subheader('Preview dos Dados')
        st.write(data)

with st.container():
    st.subheader("Histograma - Distribuição dos Gastos com Plano de Saúde", divider='rainbow' )

    # Criando um histograma 
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.hist(data["charges"], bins=50, edgecolor="black")
    ax.set_xlabel("Gastos em reais")
    ax.set_ylabel("Frequência")
    ax.set_title("Histograma dos Gastos com Plano de Saúde")
    st.pyplot(fig)

with st.container():
    st.subheader("Verificando com Boxplot se existem outliers no dataset", divider='rainbow')

    # Criando um boxplot
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.boxplot(data["charges"], vert=False)
    ax.set_yticklabels(["Gastos"])
    ax.set_title("Boxplot dos Gastos com Plano de Saúde")
    st.pyplot(fig)

    st.write("Podemos ver que existe uma certa quantidade de outliers nesse dataset, mas por hora só faremos a baseline do projeto, e refinaremos depois caso se faça necessário.")


with st.container():
    st.subheader("Histograma - Idades dos pacientes", divider='rainbow')

    # Criando um histograma 
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.hist(data["age"], bins=50, edgecolor="black")
    ax.set_xlabel("Idade")
    ax.set_ylabel("Frequência")
    ax.set_title("Histograma das idades dos pacientes")
    st.pyplot(fig)

    st.write("Notamos que existem mais pacientes entre a faixa de 49 e 50 anos")


with st.container():
    st.subheader("Histograma - Índice de massa corporal", divider='rainbow')

    # Criando um histograma 
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.hist(data["bmi"], bins=50, edgecolor="black")
    ax.set_xlabel("IMC")
    ax.set_ylabel("Frequência")
    ax.set_title("Histograma do índice de massa corporal (IMC)")
    st.pyplot(fig)

    st.write("Percebe-se um grande número de pacientes com IMC Entre 25 e 29,99, considerado acima do peso ")

with st.container():
    st.title("Distribuição de Sexo")
    def get_chart_28108359(data):
        labels = ['Masculino', 'Feminino']
        values = [data['sex'].value_counts()['male'], data['sex'].value_counts()['female']]
        custom_colors = ['#1E90FF', '#FF69B4']

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=custom_colors))])

        st.subheader(" ", divider = 'rainbow')
        st.plotly_chart(fig, theme="streamlit")

    get_chart_28108359(data)

    st.write("O gráfico acima mostra uma diferença de 6% na distribuição entre homens e mulheres no banco de dados.")


with st.container():
    st.title("Distribuição de Fumantes")
    def get_chart_28108359(data):
        labels = ['Fumante', 'Não fumante']
        values = [data['smoker'].value_counts()['yes'], data['smoker'].value_counts()['no']]
        custom_colors = ['#AAAAAA', '#FFFFFF']

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=custom_colors))])

        st.subheader(" ", divider = 'rainbow')
        st.plotly_chart(fig, theme="streamlit")

    get_chart_28108359(data)

    st.write("O gráfico acima mostra uma diferença significativa entre fumantes e não fumantes, o que pode acarretar em um aumento no valor para fumantes.")

with st.container():
    # Arredondanr a Variável AGE
    train_data['age'] = round(train_data['age'])
    
    # Transformando varáveis em numéricas
    train_data = pd.get_dummies(train_data, drop_first=True)
    
    # Escolhendo só as variáveis que vou usar
    train_data = train_data[['age', 'bmi', 'children', 'region_southeast', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'charges']]
    
    # Separando os dados
    X = train_data.iloc[:, :-1]
    y = train_data.iloc[:, -1]
    
    # Separando dados de treino e teste:
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
    
    # Arvore de regressão:
    RandomForestRegressor = RandomForestRegressor ()
    RandomForestRegressor = RandomForestRegressor.fit(X_train, y_train)
    
    # Predição:
    y_pred = RandomForestRegressor.predict(X_test)
    
    # pontuação:
    print(r2_score(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred))
    
    # Creating a pickle file for the classifier
    filename = 'predicao_plano_de_saude.pkl'
    pickle.dump(RandomForestRegressor, open(filename, 'wb'))

with st.container():
    st.title("Aplicativo de Predição de Custos de Plano de Saúde")


    # Widgets para inserção de valores
    age = st.slider("Idade", min_value=18, max_value=100, value=30)
    bmi = st.slider("Índice de Massa Corporal (IMC)", min_value=10, max_value=50, value=25)
    children = st.slider("Número de Filhos", min_value=0, max_value=10, value=0)

    # Widget para escolher o sexo
    sex_option = st.radio("Escolha o Sexo", ("Masculino", "Feminino"))

    # Widget para escolher se é fumante ou não
    smoker_option = st.radio("Fumante?", ("Sim", "Não"))

    # Widget para escolher a região
    region_option = st.selectbox("Escolha a Região", ("Sul", "Nordeste", "Sudeste", "Norte", "Centro-Oeste"))

    # Botão para fazer a predição
    if st.button("Prever"):
        
        # Transformar escolhas em valores binários
        sex_male = 1 if sex_option == "Masculino" else 0
        smoker_yes = 1 if smoker_option == "Sim" else 0
        region_southeast = 1 if region_option == "southeast" else 0
        region_northwest = 1 if region_option == "northwest" else 0
        
        input_data = [[age, bmi, children, region_southeast, sex_male, smoker_yes, region_northwest, region_southeast]]
        prediction = RandomForestRegressor.predict(input_data)[0]  

        # Formatando para resultado em português
        formatted_prediction = f"R$ {prediction:,.2f}".replace(',', '_').replace('.', ',').replace('_', '.')

        # Mostrando resultado
        st.write(f"A previsão do custo do plano de saúde é: {formatted_prediction}")
