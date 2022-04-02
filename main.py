import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title='Clustering your data', page_icon=':chart_with_upwards_trend:', layout='wide')
st.title("Clustering any DataSet you want")
st.markdown('Created by **Mohammad Bazmara**')

# ---- Functions ----
def categorize(data):
    for f_type in data.columns:
        if isinstance(data[f_type][0], str):
            data[f_type] = data[f_type].astype('category')
            data[f_type] = data[f_type].cat.codes
        else:
            pass
    return data


def visualize(data, n_clusters,x,y):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    model = KMeans(n_clusters=number_of_clusters)
    clusters = model.fit_transform(input_data)
    input_data['cluster'] = model.labels_
    for i in range(n_clusters):
        clustered_data = data[data['cluster']==i]
        plt.scatter(clustered_data[x], clustered_data[y], c=colors[i-1])
    plt.title('Clustered Database that you uploaded')
    plt.xlabel(x)
    plt.ylabel(y)
    st.pyplot()


def del_useless_features(data, number_of_instances):
    for f_type in data.columns:
        if data[f_type].nunique() == number_of_instances:
            data = data.drop(columns=[f_type])
            print('Kbooom')
    return data


# ---- Input Data ----
x = st.sidebar.file_uploader("Pick a file", type=['csv', 'xlsx'])
st.sidebar.write('[Download Sample>](https://www.kaggle.com/datasets/uciml/iris/download)')
if x is not None:
    data = pd.read_csv(x)
    number_of_instances = data.shape[1]
    input_data = data
    #input_data = del_useless_features(data, number_of_instances)
    numeric_features = list(input_data.select_dtypes(['int', 'float']).columns)
    categorize(input_data)

    # ---- SlideBar ----
    #st.sidebar.subheader('Give us your data to cluster')
    number_of_clusters = st.sidebar.slider(label='Number of clusters', min_value=2, max_value=8)
    f1 = st.sidebar.selectbox(label='X Numeric Feature:', options=numeric_features)
    f2 = st.sidebar.selectbox(label='Y Numeric Feature:', options=numeric_features)

    st.write('---')
    with st.container():
        left, right = st.columns(2)
        with left:
            st.subheader("first 5 insctances of your dataset")
            st.write(data.head(5))
        with right:
            visualize(input_data, number_of_clusters, f1, f2)
    st.write('---')