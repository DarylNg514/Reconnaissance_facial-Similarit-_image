import streamlit as st
import pandas as pd
import numpy as np
import cv2
from time import sleep
import face_recognition
import matplotlib.pyplot as plt
import streamlit as st

def choice_sidebar(option):
    if option =="Glcm":
        signatures = np.load('Signatures.npy')
        imgsimilare(signatures)
    elif option =="Haralick":
        signatures = np.load('Signatures.npy')
        imgsimilare(signatures)

    elif option =="Bit":
        signatures = np.load('Signatures.npy')
        imgsimilare(signatures)

    elif option =="Bit-Glcm":
        signatures = np.load('Signatures.npy')
        imgsimilare(signatures)

    elif option =="Bit-Haralick":
        signatures = np.load('Signatures.npy')
        imgsimilare(signatures)

    elif option =="Haralick-Glcm":
        signatures = np.load('Signatures.npy')
        imgsimilare(signatures)


def imgsimilare(signatures_class):
    X = signatures_class[:, 0: -1].astype('float')
    Y = signatures_class[:, -1]
    threshold = 0.6  # Seuil pour la distance des visages pour déterminer une correspondance
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Impossible d'ouvrir la webcam. Veuillez vérifier votre connexion.")
        return
    
    success, img = cap.read()
    if not success:
        st.error("Impossible de capturer l'image de la webcam.")
        return
    
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurrent = face_recognition.face_locations(imgS)
    
    if len(facesCurrent) == 0:
        st.error("Aucun visage détecté. Assurez-vous que votre visage est bien visible.")
        return
    
    encodesCurrent = face_recognition.face_encodings(imgS, facesCurrent)
    similar_images = []
    
    for encodeFace in encodesCurrent:
        faceDis = face_recognition.face_distance(X, encodeFace)
        matches = (faceDis <= threshold)  # Array de booléens indiquant les correspondances
        
        # Ajouter toutes les images correspondantes
        for match, name in zip(matches, Y):
            if match:
                similar_images.append(name.upper())
                
    st.image(img, channels="BGR")
    st.write("Succesfully Captured")
    
    # Affichage des images similaires
    if similar_images:
        st.write("Images similaires détectées :")
        for image_name in set(similar_images):  # Utilisation de 'set' pour éviter les doublons
            st.image(f"./images/{image_name}", caption=image_name)
    else:
        st.write("Aucune image similaire détectée.")
    
    cap.release()
          
    

def verification():
    # Importer les signatures après l'authentification
    signatures_class = np.load('FaceSignatures.npy')
    X = signatures_class[:, :-1].astype('float')
    Y = signatures_class[:, -1]
    
    # Lancer la webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Impossible d'ouvrir la webcam. Veuillez vérifier votre connexion.")
        return False

    st.write('Capturing ...')
    sleep(1)
    # Capturer une image depuis la webcam
    success, img = cap.read()

    if success:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurrent = face_recognition.face_locations(imgS)
        sleep(3)

        if len(facesCurrent) == 0:
            st.error("Aucun visage n'est détecté dans la webcam, svp veuillez bien placer votre visage et cliquer a nouveau sur le button authentifier")
            cap.release()
            return False

        encodesCurrent = face_recognition.face_encodings(imgS, facesCurrent)

        for encodeFace, faceLoc in zip(encodesCurrent, facesCurrent):
            matches = face_recognition.compare_faces(X, encodeFace)
            faceDis = face_recognition.face_distance(X, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = Y[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1 + 10, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1 + 10, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 30, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                st.image(img, channels="BGR")
                st.write("Succès de l'authentification")
                cap.release()
                sleep(3)
                return True
            else:
                name = 'Unknown'
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 30, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                st.warning("Vous n'êtes pas autorisé")
        
                cap.release()
                return False
                            
    return False
