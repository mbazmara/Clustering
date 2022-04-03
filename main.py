import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title='Clustering your data', page_icon=':chart_with_upwards_trend:')
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
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

try:
    # ---- Input Data ----
    left , right = st.columns((3,1))
    with left:
        radio_btn = ['Upload your Dataset', 'Mall Customers Dataset']
        input_data = st.radio('Select one way for clustering',radio_btn)
    with right:
        st.image('images/mmb2.jpeg', )
    if input_data == 'Upload your Dataset':
        x = st.file_uploader("Pick a file", type=['csv', 'xlsx'])
        st.write('[Download Sample>](https://www.kaggle.com/datasets/uciml/iris/download)')
        if x is not None:
            data = pd.read_csv(x)
            number_of_instances = data.shape[1]
            input_data = data
            numeric_features = list(input_data.select_dtypes(['int', 'float']).columns)
            categorize(input_data)
            # ---- SlideBar ----
            st.subheader("first 5 insctances of your dataset")
            st.write(data.head(5))
            left, middle, right = st.columns(3)
            with left:
                number_of_clusters = st.slider(label='Number of clusters', min_value=2, max_value=8)
            with middle:
                f1 = st.selectbox(label='X Numeric Feature:', options=numeric_features)
            with right:
                f2 = st.selectbox(label='Y Numeric Feature:', options=numeric_features)
            st.write('---')
            visualize(input_data, number_of_clusters, f1, f2)
            st.write('---')
    if input_data == 'Mall Customers Dataset':
        data = pd.read_csv('Mall_Customers.csv')
        number_of_instances = data.shape[1]
        input_data = data
        numeric_features = list(input_data.select_dtypes(['int', 'float']).columns)
        categorize(input_data)
        # ---- SlideBar ----
        st.subheader("first 5 insctances of your dataset")
        st.write(data.head(5))
        left, middle, right = st.columns(3)
        with left:
            number_of_clusters = st.slider(label='Number of clusters', min_value=2, max_value=8)
        with middle:
            f1 = st.selectbox(label='X Numeric Feature:', options=numeric_features)
        with right:
            f2 = st.selectbox(label='Y Numeric Feature:', options=numeric_features)
        st.write('---')
        visualize(input_data, number_of_clusters, f1, f2)
        st.write('---')
except ValueError as v:
    st.write(v)
