import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('ann_model.h5')

# Load the encoders and scaler
with open('ann_onehot_encoder_CropName.pkl', 'rb') as file:
    onehot_encoder_cropName = pickle.load(file)

with open('ann_onehot_encoder_prevCropName.pkl', 'rb') as file:
    onehot_encoder_prevCropName = pickle.load(file)

with open('ann_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Crop Rotation Recommendation')

# List of available categories for selection, removing the prefix
previous_crop_categories = [label.replace('prevCropName_', '') for label in onehot_encoder_prevCropName.categories_[0]]

# User input
Previous_Crop_Name = st.selectbox('Previous Crop Name', previous_crop_categories)
previousYield = st.slider('Previous Yield', 0.00, 100000.00, value=0.00, step=0.1)
organicCarbon = st.slider('Organic Carbon', 0.00, 100.00, value=0.00, step=0.01)
pH = st.slider('pH', 0.00, 10.00, value=0.00, step=0.01)
phosphorus = st.slider('Phosphorus', 0.00, 100.00, value=0.00, step=0.01)
cu = st.slider('Copper', 0.00, 100.00, value=0.00, step=0.01)
fe = st.slider('Ferrous', 0.00, 100.00, value=0.00, step=0.01)
potassium = st.slider('Potassium', 0.00, 100.00, value=0.00, step=0.01)
boron = st.slider('Boron', 0.00, 100.00, value=0.00, step=0.01)
zn = st.slider('Zinc', 0.00, 100.00, value=0.00, step=0.01)
sulphur = st.slider('Sulphur', 0.00, 100.00, value=0.00, step=0.01)
electricalConductivity = st.slider('Electrical Conductivity', 0.00, 100.00, value=0.00, step=0.01)
nitrogen = st.slider('Nitrogen', 0.00, 100.00, value=0.00, step=0.01)
targetYield = st.slider('Target Yield', 0.00, 100000.00, value=0.00, step=0.1)

# Encode the previous crop name
encoded_prevcrop = onehot_encoder_prevCropName.transform(np.array([Previous_Crop_Name]).reshape(-1, 1))

# Create the input data
input_data = pd.DataFrame({
    'previousYield': [previousYield],
    'organicCarbon': [organicCarbon],
    'pH': [pH],
    'phosphorus': [phosphorus],
    'cu': [cu],
    'fe': [fe],
    'potassium': [potassium],
    'boron': [boron],
    'zn': [zn],
    'sulphur': [sulphur],
    'electricalConductivity': [electricalConductivity],
    'nitrogen': [nitrogen],
    'targetYield': [targetYield]
})

# Create a DataFrame for the one-hot encoded previous crop name
encoded_df = pd.DataFrame(encoded_prevcrop, columns=onehot_encoder_prevCropName.get_feature_names_out(['previousCropName']))

# Concatenate the input data with the encoded crop name data
final_df = pd.concat([input_data, encoded_df], axis=1)

# Align the DataFrame with the expected columns for the scaler
final_df_aligned = final_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Scale the input data
expected_columns = scaler.feature_names_in_
print("Final DataFrame columns:", final_df.columns)
print("Expected columns:", expected_columns)
print("final_df_aligned ------------------------- ", final_df_aligned.columns)
input_scaled = scaler.transform(final_df_aligned.values)

# Predict the target output
prediction = model.predict(input_scaled)

# Get the predicted class
predicted_class_index = np.argmax(prediction)

# Display results
st.write(f"Predicted class index: {predicted_class_index}")
st.write(f"Predicted probability: {prediction[0][predicted_class_index]}")

# Decode the predicted label
predicted_label = onehot_encoder_cropName.categories_[0][predicted_class_index]
st.write(f"Predicted label: {predicted_label}")

