#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import sklearn 
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
    st.markdown("Streamlit web application that predicts the likelihood of bank customers subscribing to a term deposit from personal, monetary, and marketing attributes.")
    st.markdown("Developed with [this dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) from the UCI machine learning repository; based on direct marketing campaigns via phone calls of a Portuguese banking institution.")
    st.markdown("I made this app to test and highlight Streamlit as a popular web framework for building rich machine learning or deep learning applications. The layout is divided into 3 sections: one for exploratory data analysis, another for the primary classification task of a yes/no term deposit signup, and a third section for customer segmentation. Each section displays many pythonic results and allows a user to modify few components to create personalized use cases.")
    st.markdown("Insights include plots, the ability to choose models, feature importances, variable/attribute comparisons, and the like...") 
            
    tab1, tab2, tab3 = st.tabs(["EDA", "Classification and Inference", "Customer Segmentation"])
    with tab1:
        data = pd.read_csv('bank-additional-full.csv',sep = ';')
        st.subheader("The dataset at a quick glance:")
        st.markdown("Dataset:")
        st.dataframe(data, 5000, 200)
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.markdown("Correlation heatmap:")
            f, ax = plt.subplots(figsize=(10, 8))
            corr = data.corr()
            sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
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
        st.write("Any missing values?:", data.isnull().sum().any())

        st.markdown("Number of missing values per attribute:")
        st.dataframe(data.isnull().sum().to_frame(name='null count').T, 1000,50)
        #st.write(data)

        st.subheader('Attribute/feature definitions:')
        st.write("Text files containing information can be read as list(s)\
        and indexed accordingly with the dataset's column positions to display definitions or\
        suitable text of any kind.")
        

        st.subheader('Dataset descriptive statistics:')
        st.dataframe(data.describe(), 5000, 200)
        
        
        st.subheader('Attribute distribution comparison:')
        st.write("Compare a categorical and numerical attribute via boxplots and values.")
        
        #Cardinality check
        obj_list = data.loc[:, ].select_dtypes('object').columns
        obj_list_card = []
        for x in obj_list:
            obj_list_card.append(data[x].value_counts().shape[0])
        
        obj_list_new = [x for _, x in sorted(zip(obj_list_card, obj_list))]
        numerical_columns = data.select_dtypes(include=['number']).columns
        option_vald1 = st.selectbox('Attribute 1(x):', numerical_columns)
        #option_vald1 = st.selectbox('Attribute 1(x):',(data.columns))
        option_vald2 = st.selectbox('Attribute 2(y):',(obj_list_new))
        st.write('You selected:', option_vald1, "and", option_vald2)
        sns.catplot(data=data, x=option_vald1, y=option_vald2, kind="box")
        st.pyplot(plt.show())
        
        
        st.subheader('Attribute value counts:')
        option_valc = st.selectbox('Choose an attribute to see its breakdown of values by frequency.',(data.columns))
        st.dataframe(data[option_valc].value_counts().rename_axis('unique_values').reset_index(name='counts'))
    
    with tab2:
        data_classify = data.copy()
        if 'submitted' not in st.session_state:
            st.session_state.submitted = False
    
        def submit():
            st.session_state.submitted = True
    
        
        st.markdown("The main target variable ('y') is quite imbalanced, and the data itself has a fair mix of types. Gradient boosted trees can be a reliable choice.")
        st.markdown("Testing a very quick classification pipeline with two popular model choices, test set split % and even the ability to choose a custom target variable (just the categorical and binary ones for now)...")
           
        option_target = st.selectbox('Choose your target variable:', obj_list_new)
        st.write('You selected:', option_target)
    
        option_test_split = st.selectbox('Choose your test set split %:', [20, 15, 30])
        st.write('You selected:', option_test_split)
    
        option_model = st.selectbox('Choose your machine learning model for classification:', ['XGBoost', 'LightGBM'])
        st.write('You selected:', option_model)
    
        run_button = st.button("Run Classification", on_click=submit)
    
        if run_button:
            lbl0 = LabelEncoder()
            #st.write(obj_list)
            for x in obj_list_new:
                data_classify[x] = lbl0.fit_transform(data_classify[x])
               
            X = data_classify.loc[:, data_classify.columns != option_target]
            y = data_classify[option_target]
           
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = (option_test_split/100), random_state=42)    
    
            if option_model == 'LightGBM':
                def run_lightgbm():
                    lgb = lightgbm.LGBMClassifier()
                    lgb.fit(X_train, y_train)
                    y_train_preds = lgb.predict(X_train)
                    y_test_preds = lgb.predict(X_test)
        
                    explainer = shap.TreeExplainer(lgb)
                    shap_values = explainer.shap_values(X)
    
                    # explainer_heat = shap.Explainer(lgb, X)
                    # shap_values_heat = explainer_heat(X)
        
                    return accuracy_score(y_train, y_train_preds), accuracy_score(y_test, y_test_preds), y_test_preds, shap_values, lgb
        
                acc_train, acc_test, y_test_preds, shap_values, model = run_lightgbm()
        
                #LightGBM
                st.success("Fit Complete")
                st.header("Results:")
                st.write("Train accuracy:", acc_train * 100)
                st.write("Test accuracy:", acc_test * 100)
                
                st.subheader("Classification Report:")
                report = metrics.classification_report(y_test, y_test_preds, output_dict=True)
                sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
                st.pyplot(plt.show())
            
                st.subheader("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_test_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                st.pyplot(plt.show())
            
                st.subheader("Model Feature Importance:")
                lightgbm.plot_importance(model)
                plt.title("LightGBM Feature Importance (Default)")
                st.pyplot(plt.show())
            
                st.subheader("SHAP Summary Plot:")
                shap.summary_plot(shap_values, X)
                st.pyplot(plt.show())
                st.subheader("SHAP Dependence Plot For The Top 5 Features:")
                st.write("SHAP dependence plots show the effect of a single feature across the whole dataset. They plot a feature’s value vs. the SHAP value of that feature across many samples. The color corresponds to a second feature that may have an interaction effect with the feature in the plot (by default this second feature is chosen automatically). If an interaction effect is present between this second feature and the one that is opted, it will show up as a distinct vertical pattern of coloring.")
                feature_importances = model.feature_importances_               
                top_5_features_idx = feature_importances.argsort()[::-1][:5]
                top_5_feature_names = X.columns[top_5_features_idx]
                for feature_name in top_5_feature_names:
                    st.subheader(f"Dependence Plot For {feature_name}:")
                    shap.dependence_plot(feature_name, shap_values[1], X)
                    st.pyplot(plt.show())
    
            
    
            elif option_model == 'XGBoost':
                def run_xgboost():
                    xgb = xgboost.XGBClassifier()
                    xgb.fit(X_train, y_train)
                    y_train_preds = xgb.predict(X_train)
                    y_test_preds = xgb.predict(X_test)
        
                    explainer = shap.TreeExplainer(xgb)
                    shap_values = explainer.shap_values(X)
    
                    # explainer_heat = shap.Explainer(xgb, X)
                    # shap_values_heat = explainer_heat(X)
        
                    return accuracy_score(y_train, y_train_preds), accuracy_score(y_test, y_test_preds), y_test_preds, shap_values, xgb
        
                acc_train, acc_test, y_test_preds, shap_values, model = run_xgboost()
        
                st.success("Fit Complete")
                st.header("Results:")
                st.write("Train accuracy:", acc_train * 100)
                st.write("Test accuracy:", acc_test * 100)
            
                st.subheader("Classification Report:")
                report = metrics.classification_report(y_test, y_test_preds, output_dict=True)
                sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
                st.pyplot(plt.show())
            
                st.subheader("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_test_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                st.pyplot(plt.show())
            
                st.subheader("Model Feature Importance:")
                xgboost.plot_importance(model)
                plt.title("XGBoost Feature Importance (Default)")
                st.pyplot(plt.show())
            
                st.subheader("SHAP Summary Plot:")
                shap.summary_plot(shap_values, X)
                st.pyplot(plt.show())
                st.subheader("SHAP Dependence Plot For The Top 5 Features:")
                st.write("SHAP dependence plots show the effect of a single feature across the whole dataset. They plot a feature’s value vs. the SHAP value of that feature across many samples. The color corresponds to a second feature that may have an interaction effect with the feature in the plot (by default this second feature is chosen automatically). If an interaction effect is present between this second feature and the one that is opted, it will show up as a distinct vertical pattern of coloring.")
                
                feature_importances = model.feature_importances_               
                top_5_features_idx = feature_importances.argsort()[::-1][:5]
                top_5_feature_names = X.columns[top_5_features_idx]
                
                for feature_name in top_5_feature_names:
                    st.subheader(f"Dependence Plot For {feature_name}:")
                    shap.dependence_plot(feature_name, shap_values, X)
                    st.pyplot(plt.show())


                
  
    with tab3:
        st.header("Customer Segmentation")
        st.write("-> What is it? Customer segmentation is a strategy used to break up a sizable and diversified customer base into smaller\
                 groups of connected individuals who share some characteristics and are important for the marketing of\
                 a bank's offerings.")
        st.write("-> For good reason, this process is usually a top marketing focus for banks. Banks are able to provide\
                 more specialised products and services because segmentation solutions enable them to group clients based on\
                 behaviour. Additionally, marketers can maximise cross- and up-selling prospects and encourage customers\
                 to investigate similar services by better understanding client preferences.")
        st.subheader("Clustering strategy:")   
        st.write("-> When the data contains a mixture of binary and other types, a standard clustering algorithm like k-means \
        is never employed to form the respective groups.")
        st.write("-> [This great article from IBM](https://www.ibm.com/support/pages/clustering-binary-data-k-means-should-be-avoided) provides good intuition.")
        st.write("-> As a means to an end, let's try the K-modes algorithm instead. More information of which you can find in\
        [this paper](https://cse.hkust.edu.hk/~qyang/Teaching/537/Papers/huang98extensions.pdf). K-modes is designed to work with\
        continuous and categorical data formats.")
        st.write("-> As it is with most clustering initiatives, we shall break down the task into two phases; first - finding the right\
        number of clusters to form (via the elbow technique) and second, the actual clustering itself. In both phases, K-modes\
        is utilized.")
        
        st.subheader("Finding Optimal Number Of Clusters:")
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
             
        st.subheader("Clustering With K-modes:")
        option_clus = st.selectbox('How many clusters to build?', [3, 2, 4, 5])
    
        st.write('You selected:', option_clus)
        
        clusters = None  
        
        if option_clus:
            with st.spinner('Wait for it...'):
                @st.cache(persist=True)
                def run_clustering2(option_clus):
                    kmode = KModes(n_clusters=option_clus, init="random", n_init=5, verbose=2, n_jobs=-1)
                    clusters = kmode.fit_predict(dataclus)
                    return clusters
    
            clusters = run_clustering2(option_clus)
            st.success('Fit Complete!')
            st.success('Ran clustering again with ' + str(option_clus) + ' clusters!')
    
        if clusters is not None:
            dataclus.insert(0, "Cluster", clusters, True)
    
            st.subheader("Cluster Allocation:")
            st.write("The size of the formed clusters is just as important as the number of clusters themselves. \
            The following table shows the cluster breakdown of allocated data points:")
            st.write(dataclus['Cluster'].value_counts())
    
            st.subheader("Relationships Between Variables:")
            st.write("It would be interesting to observe if any two variables have a particular relationship or correlation between them as a function of clustering; i.e. a scatter plot where the color hues are set by the clusters formed.")
                
            option_x_clus = st.selectbox('Choose variable 1 (x):', data.columns)
            option_y_clus = st.selectbox('Choose variable 1 (y):', data.columns)
        
            st.write('You selected:', option_x_clus, "and", option_y_clus)
        
            scatter_plot = st.empty()
            
            with st.spinner('Creating Scatter Plot...'):
                fig, ax = plt.subplots()
                sc = ax.scatter(x=data[option_x_clus], y=data[option_y_clus], alpha=0.5,
                                marker = ".", c=dataclus['Cluster'])
                ax.legend(*sc.legend_elements(), title='Clusters')
                plt.xlabel(option_x_clus)
                plt.ylabel(option_y_clus)
                plt.title('Scatter Plot Of Selected Variables')
                scatter_plot.pyplot(plt.show())



if __name__ == '__main__':
    main()
