# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 07:33:14 2023

@author: User
"""

import cv2 as cv

# charger les classificateurs en cascade pré-entrainés
face_cascade= cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade= cv.CascadeClassifier('haarcascade_eye.xml')

# charger les images
img= cv.imread('avatar_2.png')
gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Exécution de la détection de visage
# detection de l image (image, parametre d'echelle, nbre minmum des voisins)
faces= face_cascade.detectMultiScale(gray, 1.1, 8)

# affichage des visages 
for face in faces:
    x, y, w, h = face
    
    # dessiner le rectangle sur l image principale
    cv.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), 1)
    
#Exécution de la détection de visage
# detection de l image (image, parametre d'echelle, nbre minmum des voisins)
eyes= eye_cascade.detectMultiScale(img, 1.35, 1)
for eye in eyes:
    ex, ey, ew, eh= eye
    
    # dessiner le rectangle sur l image principale
    cv.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

# affiche l'image principale
cv.imshow('image principale', img)
cv.waitKey(0)
cv.destroyAllWindows()