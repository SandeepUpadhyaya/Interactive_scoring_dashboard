import numpy as np
import pandas as pd
import pickle
import streamlit as st
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

st.title("Interactive scoring dashboard")
Scoring = st.selectbox('Scoring',('single record scoring','multiple record scoring'))
st.write('You selected:',Scoring)
st.sidebar.header('User Input Features')
st.sidebar.markdown("""
[Example CSV input file](https://github.com/SandeepUpadhyaya/New_claimrisk_MLOPS/blob/master/new_with_more_featuresformodelling.csv)
""")

if Scoring == 'single record scoring':
    Ptype = st.selectbox('Ptype',('regression','classification'))
    st.write('You selected:',Ptype)
    # Collects user input features into dataframe
    csv_file = st.sidebar.file_uploader("Upload your trained csv file(X_train) to read the feature parameters", type=["csv"])
    serialised_file = st.sidebar.file_uploader("Upload serialised file(picke or joblib file")

    loaded_model = joblib.load(serialised_file)
    features_names = pd.DataFrame(loaded_model.feature_names_in_)
    # input_df = pd.read_csv(csv_file,index_col=0)
    input_df = pd.read_csv(csv_file)
    input_data = []
    for col in input_df:
        col = st.sidebar.slider("{}".format(col),float(input_df[col].min()),float(input_df[col].max()))
        input_data.append(col)
    # st.write(input_data)   
        
    def user_input_features(input_data):
    # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = loaded_model.predict(input_data_reshaped)
    
    
        st.subheader('Prediction')
        st.write(prediction)
    
        if Ptype =='classification':
            prediction_proba = loaded_model.predict_proba(input_data_reshaped)
            st.subheader('Prediction Probability')
            st.write(prediction_proba)
            st.bar_chart(prediction_proba)
        
  

        def plot_feature_importances(model):
            feature_imp = pd.Series(model.feature_importances_,index=features_names).sort_values(ascending=False)[:10]
            st.bar_chart(feature_imp)
            st.subheader('Feature Importance')
        plot_feature_importances(loaded_model)
  
    def main():

        Prediction = user_input_features(input_data)

    if __name__ == '__main__':
        main()
            
else:
    rawcsv_multiplescoring = st.sidebar.file_uploader("Upload your input RAW CSV file for multiple scoring", type=["csv"])
    training_csv_file = st.sidebar.file_uploader("Upload your training CSV file exluding target (X_train)to define the range of values", type=["csv"])
    serialised_file = st.sidebar.file_uploader("Upload serialised file(picke or joblib file")
    # raw_df_multiplescoring = pd.read_csv(rawcsv_multiplescoring,index_col=0)
    raw_df_multiplescoring = pd.read_csv(rawcsv_multiplescoring)
    # input_df = pd.read_csv(training_csv_file,index_col=0)
    input_df = pd.read_csv(training_csv_file)
    scoring_df = raw_df_multiplescoring[raw_df_multiplescoring.columns.intersection(input_df.columns)]
    loaded_model = joblib.load(serialised_file)
# st.write(scoring_df)

# Pipe line for converting raw csv file to understable by model
# def Label_encoder(dataframe):
#         le = preprocessing.LabelEncoder()
#         for i in range(0,dataframe.shape[1]):
#             if dataframe.dtypes[i]=='object':
#                 dataframe[dataframe.columns[i]] = le.fit_transform(dataframe[dataframe.columns[i]])
#                 le_dict = dict(zip(le.classes_, le.transform(le.classes_)))


# Label_encoder(scoring_df)
    st.write(scoring_df)
    df = scoring_df
    df = df[:]
    df['prediction from model'] = loaded_model.predict(df)
    st.write(df)
    df.to_csv('result.csv')
        