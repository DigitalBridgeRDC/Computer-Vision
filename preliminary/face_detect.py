# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:42:12 2023

@author: User
"""

import cv2 as cv

# charger les classificateurs en cascade pré-entrainés
face_cascade= cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# charger les images
img= cv.imread('obama.jpg')
gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Exécution de la détection de visage
# detection de l image (image, parametre d'echelle, nbre minmum des voisins)
faces= face_cascade.detectMultiScale(gray, 1.1, 8)

# affichage des visages 
i=0
for face in faces:
    x, y, w, h = face
    
    # dessiner le rectangle sur l image principale
    cv.rectangle(img, (x,y), (x + w, y + h), (255, 0, 0 ),1)
    # extraire les visages de l'image principale
    # OpenCV et Numpy: y <-> row et x <-> col
    face = img[y:y+h, x:x+w]
    
    # afficher face0, face1, face2, etc...
    cv.imshow('face{}'.format(i), face)
    i += 1

# affiche l'image principale
cv.imshow('image principale', img)
cv.waitKey(0)
cv.destroyAllWindows()