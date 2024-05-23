import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
combat = pd.read_csv("content/combats.csv")
pokemon = pd.read_csv("content/pokemon.csv")

pokemon.head()

pokemon['Type 2'].fillna('None', inplace=True)  # Fill NaNs in 'Type 2' since many pokemon are single type

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
types = pokemon[['Type 1', 'Type 2']]  # Collect types of all pokemon
encoded_types = encoder.fit_transform(types) # Use One hot encoding for types
stats = pokemon.iloc[:,4:10].values # Collect stats of all pokemon
legendary = pokemon.iloc[:,11].astype(int).values.reshape(-1, 1) #Check legendary status
pokemon_features = np.hstack([stats, legendary, encoded_types]) # Convert collected data to fetaures

# Create a mapping from Pokémon ID to their features
pokemon_id_to_features = dict(zip(pokemon['#'], pokemon_features))
def combat_data(combat, pokemon_id_to_features):
  features, labels = [],[]

  for _,row in combat.iterrows(): #For battle
    pokemon_a_features = pokemon_id_to_features[row['First_pokemon']] # Get stats of first pokemon
    pokemon_b_features = pokemon_id_to_features[row['Second_pokemon']] # # Get stats of second pokemon
    #Convert both pokemon data to a horizontal stack and then append that horizontal stack to the features
    features.append(np.hstack([pokemon_a_features,pokemon_b_features]))
    if row['Winner'] == row['First_pokemon']:
      labels.append(0) #Return zero if first pokemon is winner
    else:
      labels.append(1) #Return 1 if second pokemon is winner
  # Return both X and Y
  return np.array(features), np.array(labels)

X, y = combat_data(combat, pokemon_id_to_features)
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform((X_train))
X_val = scaler.transform(X_val)
# Define the model
model = Sequential([
    Dense(128,activation='relu',input_shape=(X_train.shape[1],)),
    Dense(64,activation='relu'),
    Dense(1,activation='sigmoid')
])
# Compile the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

def predict_winner(Pokemon1,Pokemon2):
  Pokemon1_features = pokemon_id_to_features[Pokemon1]
  Pokemon2_features = pokemon_id_to_features[Pokemon2]
  battle_features = np.hstack([Pokemon1_features,Pokemon2_features])
  battle_features = scaler.transform([battle_features])
  prediction = model.predict(battle_features)
  if prediction < 0.5:
    return Pokemon1
  else:
    return Pokemon2
first_pokemon_id = 427 # Mega Rayquaza
second_pokemon_id = 418  # Latias

predicted_winner = predict_winner(first_pokemon_id, second_pokemon_id)
print(f'The predicted winner between Pokémon {pokemon.iloc[first_pokemon_id-1,1]} and Pokémon {pokemon.iloc[second_pokemon_id-1,1]} is Pokémon {pokemon.iloc[predicted_winner-1,1]}')
model.save('etc/my_model.keras')
joblib.dump(encoder, 'etc/one_hot_encoder.pkl')
joblib.dump(scaler, 'etc/standard_scaler.pkl')
