import streamlit as st
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the correct paths to the CSV, model, and scaler files
pokemon_file_path = os.path.join(base_dir, "content", "pokemon.csv")
model_file_path = os.path.join(base_dir, "my_model.keras")
encoder_path = os.path.join(base_dir, "one_hot_encoder.pkl")
scaler_path = os.path.join(base_dir, "standard_scaler.pkl")

# Load the CSV file
pokemon = pd.read_csv(pokemon_file_path)

# Load the model and scalers
model = load_model(model_file_path)
encoder = joblib.load(encoder_path)
scaler = joblib.load(scaler_path)

# Prepare the features for all Pokémon
types = pokemon[['Type 1', 'Type 2']]
encoded_types = encoder.transform(types)
stats = pokemon.iloc[:, 4:10].values
legendary = pokemon.iloc[:, 11].astype(int).values.reshape(-1, 1)
pokemon_features = np.hstack([stats, legendary, encoded_types])

# Create a mapping from Pokémon ID to their features
pokemon_id_to_features = dict(zip(pokemon['#'], pokemon_features))

# Function to predict
@st.cache_resource
def predict(_model, Pokemon1_id, Pokemon2_id):
    Pokemon1_features = pokemon_id_to_features[Pokemon1_id]
    Pokemon2_features = pokemon_id_to_features[Pokemon2_id]
    battle_features = np.hstack([Pokemon1_features, Pokemon2_features])
    battle_features = scaler.transform([battle_features])
    prediction = model.predict(battle_features)
    
    if prediction < 0.5:
        return pokemon.loc[pokemon['#'] == Pokemon1_id, 'Name'].values[0]
    else:
        return pokemon.loc[pokemon['#'] == Pokemon2_id, 'Name'].values[0]

# Title of the app
st.title("Pokemon Predictor")

# Instructions
st.write("Select the Pokemon you would like to battle")

# Selectbox to input Pokémon names
Pokemon_one = st.selectbox("Choose First Pokemon", pokemon['Name'])
Pokemon_two = st.selectbox("Choose Second Pokemon", pokemon['Name'])

# Get Pokémon IDs
Pokemon_one_id = int(pokemon.loc[pokemon['Name'] == Pokemon_one, '#'].values[0])
Pokemon_two_id = int(pokemon.loc[pokemon['Name'] == Pokemon_two, '#'].values[0])

# Predict and display the result
prediction = predict(model, Pokemon_one_id, Pokemon_two_id)

# Display the prediction
st.subheader("Prediction:")
st.write(f"The winner is: {prediction}")
