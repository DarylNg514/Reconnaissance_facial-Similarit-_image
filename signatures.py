import cv2
import numpy as np
import face_recognition
import os
from descriptor import glcm, bitdesc,Haralick

# Images path
path = './images'
# Global variables
images = [] # List of images
classNames = [] # List of image names
# Grab all images from the folder
myList = os.listdir(path)

# Load images
for img in myList:
    curImg = cv2.imread(os.path.join(path, img))
    images.append(curImg)
    
    # Obtenez le nom de fichier complet avec extension
    imgName = os.path.splitext(img)[0]
    imgExtension = os.path.splitext(img)[1]
    
    classNames.append(imgName + imgExtension)  # Ajoutez le nom de fichier avec extension

def extract_features(img):
    """_summary_
    Extract features from a grayscale image
    Descriptors: GLCM, BiT, Haralick, etc,

    Args:
        image_path (_type_): Provide the relative path of the query image
    """
    features_glcm=glcm(img)

    features_bit=bitdesc(img)

    features_haralick=Haralick(img)

    features_gl=glcm(img)
    features_hara1 = Haralick(img)
    features_glhara = features_hara1 + features_gl

    features_bi=bitdesc(img)
    features_hara2 = Haralick(img)
    features_bihara = features_hara2 + features_bi 

    features_b=bitdesc(img)
    features_g = glcm(img)
    features_bigl = features_b + features_g 

    return features_glcm,features_bit,features_haralick,features_glhara,features_bihara,features_bigl



# Define find face and encode function
def findEncodings(img_List, imgName_List):
    """_summary_

    Args:
        img_List (_type_): _description_
        imgName_List (_type_): _description_
    """
    all_features_glcm = []  # Liste pour stocker toutes les caractéristiques GLCM
    all_features_bit = []  # Liste pour stocker toutes les caractéristiques BiT
    all_features_haralick = []  # Liste pour stocker toutes les caractéristiques Haralick
    all_features_glhara = []  # Liste pour stocker toutes les caractéristiques GLCM + Haralick
    all_features_bithara = []  # Liste pour stocker toutes les caractéristiques BiT + Haralick
    all_features_bitglcm = []  # Liste pour stocker toutes les caractéristiques BiT + GLCM

    count = 1
    for myImg, name in zip(img_List, imgName_List):
        features_glcm, features_bit, features_haralick, features_glhara, features_bithara, features_bigl = extract_features(myImg)  # Extract features from the image
        
        img = cv2.cvtColor(myImg, cv2.COLOR_BGR2RGB)
        signature = face_recognition.face_encodings(img)[0]

        # Créer des classes de signature avec le nom de l'image
        signature_class = signature.tolist() + [name]

        # Ajouter les caractéristiques extraites à chaque classe de signature
        features_glcm.append(signature_class.copy())
        features_bit.append(signature_class.copy())
        features_haralick.append(signature_class.copy())
        features_glhara.append(signature_class.copy())
        features_bithara.append(signature_class.copy())
        features_bigl.append(signature_class.copy())
        print(features_glcm)
        all_features_glcm.append(features_glcm)
        all_features_bit.append(features_bit)
        all_features_haralick.append(features_haralick)
        all_features_glhara.append(features_glhara)
        all_features_bithara.append(features_bithara)
        all_features_bitglcm.append(features_bigl)

        print(f'{int((count / len(img_List)) * 100)} % extracted ...')
        count += 1

    # Convertir toutes les listes de caractéristiques en tableaux numpy et les sauvegarder
    np.save('signatures_glcm.npy', np.array(all_features_glcm, dtype=object))
    np.save('signatures_bit.npy', np.array(all_features_bit, dtype=object))
    np.save('signatures_haralick.npy', np.array(all_features_haralick, dtype=object))
    np.save('signatures_glhara.npy', np.array(all_features_glhara, dtype=object))
    np.save('signatures_bithara.npy', np.array(all_features_bithara, dtype=object))
    np.save('signatures_bitglcm.npy', np.array(all_features_bitglcm, dtype=object))

    print('Data stored successfully!')


def main():
    findEncodings(images, classNames)
    

if __name__ == '__main__':
    main()


'''import cv2
import numpy as np
import face_recognition
import os
from data_processing import extract_features

# Images path
path = './images'
# Global variables
images = [] # List of images
classNames = [] # List of image names
# Grab all images from the folder
myList = os.listdir(path)
#print(myList)
# Load images
for img in myList:
    curImg = cv2.imread(os.path.join(path, img))
    images.append(curImg)
    
    # Obtenez le nom de fichier complet avec extension
    imgName = os.path.splitext(img)[0]
    imgExtension = os.path.splitext(img)[1]
    
    classNames.append(imgName + imgExtension)  # Ajoutez le nom de fichier avec extension

# Define find face and encode function
def findEncodings(img_List, imgName_List):
    """_summary_

    Args:
        img_List (_type_): _description_
        imgName_List (_type_): _description_
    """
    signatures = []
    count = 1
    for myImg, name in zip(img_List, imgName_List):
        img = cv2.cvtColor(myImg, cv2.COLOR_BGR2RGB)
        signature = face_recognition.face_encodings(img)[0]
        signature_class = signature+ [name]
        signatures.append(signature_class)
        print(extract_features(signature_class))
        print(signature_class)
        print(f'{int((count/(len(img_List)))*100)} % extracted ...')
        count += 1
    face_array = np.array(signatures)
    np.save('Signatures.npy', face_array)
    print('Signature saved')

def main():
    findEncodings(images, classNames)

if __name__ == '__main__':
    main()
'''    