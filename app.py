import streamlit as st
from ultralytics import YOLO
from playsound3 import playsound

# Chargement du modèle
model = YOLO("/home/wecode-063/Rendu_Data/C-DAT-900-ABJ-2-2-ecp-joseph.m-baye/best.pt")

# Streamlit pour l'interface utilisateur
st.title("Détection d'Accidents avec YOLO")

uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "png"])

if uploaded_file:
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image("uploaded_image.jpg", caption="Image Téléchargée", use_container_width=True)

    # Prédiction
    alerte = model.predict(
        source="uploaded_image.jpg", imgsz=640, conf=0.35, save=True, show=True
    )
    sound = "Sound/UNE MENACE A ETE DETECTEE VIDEO OFFICIELLES.mp3"
    
    for alert in alerte:
        if alert.boxes:
            st.error("Accident détecté !")
            playsound(sound)
            break
        else:
            st.success("Aucun accident détecté.")

    # Supprimer le fichier temporaire
