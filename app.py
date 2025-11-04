import streamlit as st
import pandas as pd
import pickle

# Load the models and scaler
scaler = pickle.load(open("scaler.pkl", 'rb'))
model_gbc = pickle.load(open("model_gbc.pkl", 'rb'))

def predict_chronic_disease(age, bp, sg, al, hemo, sc, htn, dm, cad, appet, pc):
    df_dict = {
        'age': [age],
        'bp': [bp],
        'sg': [sg],
        'al': [al],
        'hemo': [hemo],
        'sc': [sc],
        'htn': [htn],
        'dm': [dm],
        'cad': [cad],
        'appet': [appet],
        'pc': [pc]
    }

    df = pd.DataFrame(df_dict)

    # Fill missing numeric values before encoding/scaling
    df[['age', 'bp', 'sg', 'al', 'hemo', 'sc']] = df[['age', 'bp', 'sg', 'al', 'hemo', 'sc']].apply(pd.to_numeric, errors='coerce')
    df[['age', 'bp', 'sg', 'al', 'hemo', 'sc']] = df[['age', 'bp', 'sg', 'al', 'hemo', 'sc']].fillna(df[['age', 'bp', 'sg', 'al', 'hemo', 'sc']].mean(numeric_only=True))

    # Fill missing categorical values
    df = df.fillna({
        'htn': 'no',
        'dm': 'no',
        'cad': 'no',
        'appet': 'good',
        'pc': 'normal'
    })

    # Clean text inputs (handle capitalization and unexpected inputs)
    df['htn'] = df['htn'].astype(str).str.lower().str.strip()
    df['dm'] = df['dm'].astype(str).str.lower().str.strip()
    df['cad'] = df['cad'].astype(str).str.lower().str.strip()
    df['appet'] = df['appet'].astype(str).str.lower().str.strip()
    df['pc'] = df['pc'].astype(str).str.lower().str.strip()

    # Fill missing or invalid categorical values
    df = df.fillna({
        'htn': 'no',
        'dm': 'no',
        'cad': 'no',
        'appet': 'good',
        'pc': 'normal'
    })

    # Map categorical values
    df['htn'] = df['htn'].map({'yes': 1, 'no': 0}).fillna(0)
    df['dm'] = df['dm'].map({'yes': 1, 'no': 0}).fillna(0)
    df['cad'] = df['cad'].map({'yes': 1, 'no': 0}).fillna(0)
    df['appet'] = df['appet'].map({'good': 1, 'poor': 0}).fillna(1)
    df['pc'] = df['pc'].map({'normal': 1, 'abnormal': 0}).fillna(1)


    # Scale numeric columns (ensure they are numeric)
    numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # print("NaN check before prediction:\n", df.isna().sum())

    # Make the prediction
    prediction = model_gbc.predict(df)

    return prediction[0]



# Streamlit UI
st.title("Chronic Kidney Disease Prediction")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    bp = st.number_input("Blood Pressure", min_value=10, max_value=200, value=80)
    sg = st.number_input("Specific Gravity", min_value=1.005, max_value=1.050, value=1.020)
    al = st.number_input("Albumin", min_value=0.0, max_value=5.0, value=1.0)
    hemo = st.number_input("Hemoglobin", min_value=5.0, max_value=20.0, value=15.4)
    sc = st.number_input("Serum Creatinine", min_value=0.5, max_value=10.0, value=1.2)

with col2:
    # Dropdown for condtions
    htn = st.selectbox("Hypertension", ['yes', 'no'])
    dm = st.selectbox("Diabetes", ['yes', 'no'])
    cad = st.selectbox("Coronary Artery Disease", ['yes', 'no'])
    appet = st.selectbox("Appetite", ['yes', 'no'])
    pc = st.selectbox("Protien in Urine", ['yes', 'no'])

# When the user clicks the "predict button"

if st.button("Predict"):
    # Make the prediction
    result = predict_chronic_disease(age,bp,sg,al,hemo,sc,htn,dm,cad,appet,pc)

    # Display the result
    if result == 1:
        st.write("### The patient has Chronic Kidney Disease (CKD).")
    else:
        st.write("### The patient does not have Chronic Kidney Disease (CKD).")