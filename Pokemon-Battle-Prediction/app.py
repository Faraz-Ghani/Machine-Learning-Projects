import streamlit as st
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

model = load_model('my_model.h5')
pokemon = pd.read_csv("/content/Pokemon/pokemon.csv")

# Function to predict
@st.cache
def predict(model, Pokemon1_id, Pokemon2_id):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    types = pokemon[['Type 1', 'Type 2']]  # Collect types of all pokemon
    encoded_types = encoder.fit_transform(types)  # Use One hot encoding for types
    stats = pokemon.iloc[:, 4:10].values  # Collect stats of all pokemon
    legendary = pokemon.iloc[:, 11].astype(int).values.reshape(-1, 1)  # Check legendary status
    pokemon_features = np.hstack([stats, legendary, encoded_types])  # Convert collected data to features

    # Create a mapping from Pokémon ID to their features
    pokemon_id_to_features = dict(zip(pokemon['#'], pokemon_features))
    Pokemon1_features = pokemon_id_to_features[Pokemon1_id]
    Pokemon2_features = pokemon_id_to_features[Pokemon2_id]
    battle_features = np.hstack([Pokemon1_features, Pokemon2_features])
    scaler = StandardScaler()
    battle_features = scaler.fit_transform([battle_features])
    prediction = model.predict(battle_features)
    
    if prediction < 0.5:
        return pokemon.loc[pokemon['#'] == Pokemon1_id, 'Name'].values[0]
    else:
        return pokemon.loc[pokemon['#'] == Pokemon2_id, 'Name'].values[0]

# Title of the app
st.title("Pokemon Predictor")

# Instructions
st.write("""
Select the Pokemon you would like to battle
""")

# Selectbox to input Pokémon names
Pokemon_one = st.selectbox("Choose First Pokemon", pokemon['Name'])
Pokemon_two = st.selectbox("Choose Second Pokemon", pokemon['Name'])

# Get Pokémon IDs
Pokemon_one_id = int(pokemon.loc[pokemon['Name'] == Pokemon_one, '#'].values[0])
Pokemon_two_id = int(pokemon.loc[pokemon['Name'] == Pokemon_two, '#'].values[0])


prediction = predict(model, Pokemon_one_id, Pokemon_two_id)

# Display the prediction
st.subheader("Prediction:")
st.write(f"The winner is: {prediction}")
