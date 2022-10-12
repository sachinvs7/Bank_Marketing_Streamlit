#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost
import lightgbm
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pylab as plt
import seaborn as sns

shap.initjs()


def main():
    
    st.title("Bank Marketing Analysis")
    st.markdown("Streamlit web application that predicts the likelihood of bank customers subscribing to a term deposit from personal, monetary, and marketing features.")
    st.markdown("Developed with [this dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) from the UCI machine learning repository; based on direct marketing campaigns via phone calls of a Portuguese banking institution.")
    st.markdown("I made this app to test and highlight Streamlit as a popular web framework for building rich machine or deep learning applications. The layout is divided into 3 sections, one for exploratory data analysis, another for the primary classification task of a yes/no term deposit signup and third - clustering for customer segmentation. Each section shows many pythonic results and allows a user to filter data and modify few components to create a brief 'personalized' use case. Or atleast that is the intention so that the processes defined behind the scenes in code can still work even if the dataset is swapped.")
    st.markdown("Insights include plots, the ability to choose models, feature importances, variable/attribute comparisons and the like...") 
            
    tab1, tab2, tab3 = st.tabs(["EDA", "Classification and Inference", "Customer Segmentation"])
    with tab1:
        data = pd.read_csv('bank-additional-full.csv',sep = ';')
        st.subheader("The dataset at a quick glance")
        st.write("After a field/entry for uploading data, a snapshot of information from the dataset can be presented immediately - such as size, type distributions, correlation heatmap, etc. Not ordered by relevance here. Testing aesthetics...")
        st.markdown("Dataset:")
        st.dataframe(data, 5000, 200)
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown("Correlation Heatmap:")
            f, ax = plt.subplots(figsize=(10, 8))
            corr = data.corr()
            sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                        square=True, ax=ax)
            st.pyplot(plt.show())
            st.write("Dataset shape:", data.shape)
            st.write()
            st.write("Number of duplicate records:", data.duplicated().sum())
            

        with col2:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.markdown("Dataset type distribution:")
            plt.figure(figsize=(7, 5.85))
            ax = data.dtypes.to_frame().value_counts().plot(kind='bar', rot=0)
            ax.set_title("Distribution Of Data Types", y = 1)
            ax.set_xlabel('Type')
            ax.set_ylabel('Number Of Attributes')
            for rect in ax.patches:
                y_value = rect.get_height()
                x_value = rect.get_x() + rect.get_width() / 2
                space = 1
                label = "{:.0f}".format(y_value)
                ax.annotate(label, (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center', va='bottom')
            st.pyplot(plt.show())
            
        if data.isnull().sum().sum() == 0:
            tot_miss = 0
        st.write("Any missing values?", data.isnull().sum().any())

        st.markdown("Number of missing values per attribute:")
        st.dataframe(data.isnull().sum().to_frame(name='null count').T, 1000,50)
        #st.write(data)

        st.subheader('Attribute Definitions')
        st.write("Text files containing information can be read as list(s)\
        and indexed accordingly with the dataset's column positions to display definitions or\
        suitable text of any kind.")
        

        st.subheader('Dataset Descriptive Statistics')
        st.dataframe(data.describe(), 5000, 200)
        
        
        st.subheader('Attribute Distribution Comparisons')
        st.write("Compare a categorical and numerical attribute via boxplots and values.")
        option_vald1 = st.selectbox('Attribute 1(x):',(data.columns))
        option_vald2 = st.selectbox('Attribute 2(y):',(data.columns))
        st.write('You selected:', option_vald1, "and", option_vald2)
        sns.catplot(data=data, x=option_vald1, y=option_vald2, kind="box")
        st.pyplot(plt.show())
        
        
        st.subheader('Attribute Value Counts')
        option_valc = st.selectbox('Choose an attribute to see its breakdown of values by frequency.',(data.columns))
        st.dataframe(data[option_valc].value_counts().rename_axis('unique_values').reset_index(name='counts'))
    
    with tab2:
        st.markdown("Testing a very quick classification pipeline with two popular choices...")
        st.markdown("The desired target variable is quite imbalanced, and the data itself has a fair mix of types. Hence gradient boosted trees can be a reliable choice.")
        st.markdown("There are several steps one can employ to build a coherent thought out model\
        but for now, simple type conversions and one-hot encoding the categorical variables would do.")
        option_model = st.selectbox('Choose your machine learning model for classification:',(['XGBoost', 'LightGBM']))
        st.write('You selected:', option_model)
        
        #@st.cache(persist=True)
        option_target = st.selectbox('Choose your target variable:',(data.columns))
        st.write('You selected:', option_target)
        
        option_test_split = st.selectbox('Choose your test split %:',([20, 15, 30]))
        st.write('You selected:', option_test_split)
        
        data_classify = data.copy()
        obj_list = data_classify.loc[:, data_classify.columns != option_target].select_dtypes('object').columns
        lbl0 = preprocessing.LabelEncoder()
        for x in obj_list:
            data_classify[x] = lbl0.fit_transform(data_classify[x])


        X = data_classify.loc[:, data_classify.columns != option_target]
        y = data_classify[option_target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

        lbl1 = preprocessing.LabelEncoder()
        y_train = lbl1.fit_transform(y_train)
        y_test = lbl1.transform(y_test)
        
        
        if option_model == 'LightGBM':
            with st.spinner('Wait for it...'):
                @st.experimental_singleton
                def run_lightgbm():
                    
                    lgb = lightgbm.LGBMClassifier()
                    lgb.fit(X_train,y_train)
                    y_train_preds = lgb.predict(X_train)
                    y_test_preds = lgb.predict(X_test)
                    
                    explainer = shap.TreeExplainer(lgb)
                    shap_values = explainer.shap_values(X)
                    
                    return accuracy_score(y_train,y_train_preds),accuracy_score(y_test,y_test_preds),y_test_preds,shap_values,lgb
            acc_train, acc_test, y_test_preds, shap_values, model = run_lightgbm()
        
        
        
        if option_model == 'XGBoost':
            with st.spinner('Wait for it...'):
                @st.experimental_singleton
                def run_xgboost():
                    
                    xgb = xgboost.XGBClassifier()
                    xgb.fit(X_train,y_train)
                    y_train_preds = xgb.predict(X_train)
                    y_test_preds = xgb.predict(X_test)
                    
                    explainer = shap.TreeExplainer(xgb)
                    shap_values = explainer.shap_values(X)
                    
                    return accuracy_score(y_train,y_train_preds),accuracy_score(y_test,y_test_preds),y_test_preds,shap_values,xgb
            acc_train, acc_test, y_test_preds, shap_values, model = run_xgboost()
                

        st.success("Fit Complete")
        st.header("Results")
        st.write("Train accuracy:", acc_train*100) 
        st.write("Test accuracy:", acc_test*100)
        st.subheader("Classification Report")

        report = metrics.classification_report(y_test, y_test_preds, output_dict=True)
        sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
        st.pyplot(plt.show())
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_test_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        st.pyplot(plt.show())
        st.subheader("Feature Importance")
        if option_model == 'XGBoost':
            xgboost.plot_importance(model)
            plt.title("XGBoost Feature Importance (Default)")
            st.pyplot(plt.show())
            st.subheader("SHAP Summary Plot")
            shap.summary_plot(shap_values, X)
            st.pyplot(plt.show())
            st.subheader("Relationships Between Variables")
            st.write("It would be interesting to observe if any two variables have a \
            particular relationship or correlation between them. Select them below for a scatter plot as a function of clustering; \
            i.e color hues set by the clusters formed.")
            option_shap_var = st.selectbox('Choose variable for dependence plot',(X.columns))
            shap.dependence_plot(option_shap_var, shap_values, X)
            st.pyplot(plt.show())
            
        if option_model == 'LightGBM':
            lightgbm.plot_importance(model)
            plt.title("LightGBM Feature Importance (Default)")
            st.pyplot(plt.show())
            st.subheader("SHAP Summary Plot")
            shap.summary_plot(shap_values, X)
            st.pyplot(plt.show())
            st.subheader("SHAP Dependence Plot")
            st.write("SHAP dependence plots show the effect of a single feature across the whole dataset. They plot a feature’s value vs. the SHAP value of that feature across many samples. The color corresponds to a second feature that may have an interaction effect with the feature we are plotting (by default this second feature is chosen automatically). If an interaction effect is present between this second feature and the one we are plotting it will show up as a distinct vertical pattern of coloring. ")
            option_shap_var = st.selectbox('Choose variable for dependence plot',(X.columns))
            shap.dependence_plot(option_shap_var, shap_values[1], X)
            st.pyplot(plt.show())
        
        


    
    
    with tab3:
        st.header("Customer Segmentation")
        st.write("-> What is it? Customer segmentation is a strategy used to break up a sizable and diversified customer base into smaller\
                 groups of connected individuals who share some characteristics and are important for the marketing of\
                 a bank's offerings.")
        st.write("-> Why do it? For good reason, this process is usually a top marketing focus for banks. Banks are able to provide\
                 more specialised products and services because segmentation solutions enable them to group clients based on\
                 behaviour. Additionally, marketers can maximise cross- and up-selling prospects and encourage customers\
                 to investigate similar services by better understanding client preferences.")
        st.subheader("Clustering strategy")   
        st.write("-> When the data contains a mixture of binary and other types, a standard clustering algorithm like k-means \
        is never employed to form the respective groups.")
        st.write("-> The reasoning behind that avoidance is beyond the scope of this project where there focus is on Streamlit\
        but I would encourage you to check out [this great article from IBM](https://www.ibm.com/support/pages/clustering-binary-data-k-means-should-be-avoided) for the entire intuition.")
        st.write("-> As a means to an end, we shall be using the k-modes algorithm instead. More information of which you can find in\
        [this paper](https://cse.hkust.edu.hk/~qyang/Teaching/537/Papers/huang98extensions.pdf). K-modes is designed to work with\
        continuous and categorical data formats.")
        st.write("-> As it is with most clustering initiatives, we shall break down the task into two phases; first - finding the right\
        number of clusters to form (via the elbow technique) and second, the actual clustering itself. In both phases, k-modes\
        is utilized.")
        
        st.subheader("Finding Optimal Number Of Clusters")
        from kmodes.kmodes import KModes
        import time
        dataclus = data.copy()
        
        with st.spinner('Wait for it...'):
            @st.cache(persist=True)
            def run_clustering():
                cost = []
                K = range(1,5)
                for num_clusters in list(K):
                    kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 5, verbose=2, n_jobs=-1)
                    kmode.fit_predict(dataclus)
                    cost.append(kmode.cost_)

                return K, cost
                
        K, cost = run_clustering()
        st.success('Done!')
        st.write("Lowest cost achieved:", min(cost))
        st.write("Number of clusters that achieved the lowest cost:", cost.index(min(cost)))
        st.write("Recommended optimal number of clusters:", cost.index(min(cost)))
        plt.plot(K, cost, 'bx-')
        plt.xlabel('Number Of Clusters')
        plt.ylabel('Cost')
        plt.title('Elbow Method For Optimal k')
        st.pyplot(plt.show())
        
        
        
        st.subheader("Clustering with k-modes")
        option_clus = st.selectbox('How many clusters to build?',([3,2,4,5]))

        st.write('You selected:', option_clus)
        with st.spinner('Wait for it...'):
            @st.cache(persist=True)
            def run_clustering2():
                kmode = KModes(n_clusters=3, init = "random", n_init = 5, verbose=2, n_jobs=-1)
                clusters = kmode.fit_predict(dataclus)
                return clusters
        st.success('Fit Complete!')    
        clusters = run_clustering2()
        

        
            
        dataclus.insert(0, "Cluster", clusters, True)
        
        st.subheader("Cluster Allocation")
        st.write("The size of the formed clusters is just as important as the number of clusters themselves. \
        The following table shows the cluster breakdown of allocated data points:")
        st.write(dataclus['Cluster'].value_counts())
        
        st.subheader("Relationships Between Variables")
        st.write("It would be interesting to observe if any two variables have a \
        particular relationship or correlation between them as a function of clustering; i.e scatter plot  \
        where the color hues are set by the clusters formed.")
        
 
        option_x_clus = st.selectbox('Choose variable 1 (x)',(data.columns))
        option_y_clus = st.selectbox('Choose variable 1 (y)',(data.columns))

        st.write('You selected:', option_x_clus, "and", option_y_clus)

        #import seaborn as sns
        #st.pyplot(sns.relplot(x=option_x_clus, y=option_y_clus, hue='Cluster', data=dataclus))
        
        fig, ax = plt.subplots()
        sc = ax.scatter(x=data[option_x_clus], y=data[option_y_clus], alpha=0.5,c=dataclus['Cluster'])
        ax.legend(*sc.legend_elements(), title='Clusters')
        plt.xlabel(option_x_clus)
        plt.ylabel(option_y_clus)
        plt.title('Scatter Plot Of Selected Variables')
        st.pyplot(plt.show())


        

        
        


if __name__ == '__main__':
    main()
