import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings



#https://www.kaggle.com/code/methoomirza/crop-visualization-and-prediction-ml

st.beta_set_page_config(page_title="Make Farming Recommender", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

# Barplot Function for Compasison Graph
def crop_relation_visual(yfeature,df):
    ax = sns.set_style('whitegrid')
    plt.subplots(figsize=(15,8))

    ax = sns.barplot(x="label", y=yfeature, data=df, ci=None)
    #ax.bar_label(ax.containers[0], fontsize=12)

    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.title("Crops Relation with " + str(yfeature), fontsize = 24)
    plt.xlabel("Crops Name", fontsize = 18)
    plt.ylabel("values of " + str(yfeature), fontsize = 18)


def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Make Farming Recommendation  🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    test1=[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
    test2= pd.DataFrame(test1,columns= ['apple','banana','blackgram','chickpea','coconut','coffee','cotton','grapes','jute','kidneybeans','lentil','maize','mango','mothbeans','mungbean','muskmelon','orange','papaya','pigeonpeas','pomegranate','rice','watermelon'],dtype=int)
   # st.table(test2)
    #st.write(    (test2 == 2).idxmax(axis=1)[0])

    col1,col2  = st.beta_columns([2,2])
   # df = pd.read_csv('Crop_recommendation.csv')
    selected_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    #for x in selected_features:
        #crop_relation_visual(x,df)
    
    with col2: 
        with st.beta_expander(" ℹ️ Information", expanded=False):
            st.write("""
            Make Farming recommendation is one of the most important aspects of precision agriculture. Make Farming recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address Make Farming selection issues.  
           

        ## How does it work ❓ 
        Complete all the parameters and the machine learning model will predict the most suitable Make Farmings to grow in a particular farm based on various parameters
       
            """)
       


    with col1:
        st.subheader(" Find out the most suitable Make Farming to grow in your farm 👨‍🌾")
        N = st.number_input("Nitrogen", 1,10000)
        P = st.number_input("Phosporus", 1,10000)
        K = st.number_input("Potassium", 1,10000)
        temp = st.number_input("Temperature",0.0,100000.0)
        humidity = st.number_input("Humidity in %", 0.0,100000.0)
        ph = st.number_input("Ph", 0.0,100000.0)
        rainfall = st.number_input("Rainfall in mm",0.0,100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):

            loaded_model = load_model('model.pkl')
            prediction = loaded_model.predict(single_pred)
            col1.write('''
		    ## Results 🔍 
		    ''')
            #col1.success(f"{prediction.item()} are recommended by the A.I for your farm.")
            #col1.write(f"{prediction.item()}")
            col1.success(f"{(test2 == prediction.item()).idxmax(axis=1)[0]} are recommended by the A.I for your farm.")
      #code for html ☘️ 🌾 🌳 👨‍🌾  🍃

    #st.warning("Note: This A.I application is for educational/demo purposes only and cannot be relied upon. Check the source code [here](https://github.com/gabbygab1233/Make Farming-Recommendation)")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()
