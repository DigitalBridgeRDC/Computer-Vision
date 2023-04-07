# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 07:56:10 2023

@author: User
"""

import cv2 as cv
import sys

# charger le classificateur en cascade pré-entrainé
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# charger l'image
img = cv.imread('dan.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# exécuter la détection de visage
faces = face_cascade.detectMultiScale(img_gray, 1.1, 8)

# vérifier le nombre de visages
if len(faces) != 2:
    sys.exit('la photo doit avoir exactement 2 visages, réessayez...')
    
# récupération des dimensions de chaque visage
x1, y1, w1, h1 = faces[0]
x2, y2, w2, h2 = faces[1]

# extraction des 2 visages de l'image
face1 = img[y1:y1+h1,x1:x1+w1]
face2 = img[y2:y2+h2,x2:x2+w2]

# redimensionner face2 aux dimensions de face1 et vice versa
face2 = cv.resize(face2,(w1,h1))
face1 = cv.resize(face1,(w2,h2))

# remplacer face2 par face1
img[y2:y2+h2,x2:x2+w2] = face1

# remplacer face1 par face2
img[y1:y1+h1,x1:x1+w1] = face2

# afficher l'échange de visages
cv.imshow('échange', img)
cv.waitKey(0)
cv.destroyAllWindows()