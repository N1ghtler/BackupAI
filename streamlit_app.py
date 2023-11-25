from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

from joblib import load

img_width = 64
img_height = 64

# load 
model = load('fc_svm_eigenfaces_model_20.joblib')
label_dic = pd.read_csv("class_names_20.csv")
loaded_lfw = np.load('lfw_data_20.npz')

# Extract the arrays
loaded_data = loaded_lfw['data']
loaded_labels = loaded_lfw['labels']

X_train, X_test, y_train, y_test = train_test_split(loaded_data, loaded_labels, test_size=0.4, shuffle=True, stratify=loaded_labels, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

h, w = 64, 64
n_components = 500 #192

pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

def predict_image(img):
    img = np.array(img)
    img = resize(img, (img_width, img_height))
    img = img.flatten()
    img = img.reshape(1, -1)
    img = scaler.transform(img)

    # Make a prediction using the model
    img_pca = pca.transform(img)
    predicted_prob = model.predict_proba(img_pca)
    predicted_class = model.predict(img_pca)

    top_5 = np.argsort(predicted_prob[0])[-62:]
    top_5 = top_5[::-1]

    name = []
    name_per = []

    for i in range(len(top_5)):
        name.append(label_dic.loc[top_5[i], "class"])
        name_per.append(str(round(predicted_prob[0][top_5[i]]*100)) + "%")

    return label_dic.loc[predicted_class[0], "class"], predicted_prob[0][predicted_class[0]] * 100, name, name_per

st.title('Image Classification')

uploaded_file = st.file_uploader("Choose an image...",  type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label, prob, name, name_per = predict_image(image)
    st.write(f'Class: {name[0]}, Probability: {name_per[0]}')
    st.write('Top 3 Others Classes:')
    st.write(f'Classes: {name[1]}, Probabilities: {name_per[1]}')
    st.write(f'Classes: {name[2]}, Probabilities: {name_per[2]}')
    st.write(f'Classes: {name[3]}, Probabilities: {name_per[3]}')
