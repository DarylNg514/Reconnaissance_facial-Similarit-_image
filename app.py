import numpy as np
import streamlit as st
from Methods import choice_sidebar, verification
from PIL import Image

# Utiliser une variable de session pour suivre l'état de l'authentification
auth_done = st.session_state.get('auth_done', False)

def main():
    st.write(
        """

        # Content-Based Image Retrieval(CBIR): La Recherche d'Images par le Contenu 
        ###### Trouver des Images Similaires à Travers le Prisme de la Vision par Ordinateur

        """
    )

    # Si l'authentification n'a pas déjà été effectuée, afficher le bouton d'authentification
    if not auth_done:
        if st.button("S'authentifier"):
            # Effectuer la vérification
            if verification():
                # Marquer l'authentification comme réussie dans la variable de session
                st.session_state.auth_done = True
                st.experimental_rerun()
            else:
                st.warning("Unauthorized")
        return
    
    # Si l'authentification a déjà été effectuée, afficher le sélecteur de descripteur
    option_selected = st.sidebar.selectbox(
        "Choisir un Descripteur :",
        ("Glcm", "Haralick", "Bit", "Bit-Glcm", "Bit-Haralick", "Haralick-Glcm"), None
    )

    if option_selected:
        if(st.button("Commencer")):
            choice_sidebar(option_selected)
    else:
        st.warning("Veuillez choisir le descripteur svp..")

if __name__ == '__main__':
    main()
