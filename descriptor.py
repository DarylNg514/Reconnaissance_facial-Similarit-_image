from skimage.feature import graycomatrix, graycoprops
import cv2
from BiT import bio_taxo
from mahotas import features


def glcm(data):
    if len(data.shape) > 2:
        gray_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = data
    co_matrix= graycomatrix(gray_image, [2], [0], None, symmetric=True, normed=True)
    diss = graycoprops(co_matrix, 'dissimilarity' )[0, 0]
    cont = graycoprops(co_matrix,'contrast' )[0, 0]
    corr = graycoprops(co_matrix, 'correlation')[0, 0]
    ener = graycoprops(co_matrix, 'energy')[0, 0]
    homo = graycoprops(co_matrix, 'homogeneity')[0, 0]
    return [diss, cont, corr, ener, homo]

def bitdesc(data):
    if len(data.shape) > 2:
        gray_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = data
    return bio_taxo(gray_image)

def Haralick(data):
    if len(data.shape) > 2:
        gray_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = data
    haralick_features = features.haralick(gray_image).mean(axis=0)
    return haralick_features.tolist()
