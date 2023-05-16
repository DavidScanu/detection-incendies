# Detection incendies

*Geoffroy Daumer, Amadou Diaby, David Scanu*

---

Notebook pour l'entrainement du modèle : https://colab.research.google.com/drive/1QbW2wbWThUDJmFHIMrQBIWmWU0s9UtWN

- Labélisation des données
- Visualisation des données
- Choix du modèle
- Entrainement du modèle (avec couches gelées)
- Tests de prédictions

---

## Améliorations à venir : 

- Implémentation de la détection sur les **videos**
- Implémentation de la détection sur la **webcam**
- Meilleur modèle (plus léger, plus rapide)
- Implémentation de Torch pour les prédictions
- "filename" du fichier d'origine sauvegarder dans le `document` MongoBD
- Meilleures maitrise du `Results object`
- CRUD complet MongoDB + fichier sauvegardés (Effacer les détections passées + les fichiers images sauvegardés)
- Grille photos pour les détections passées

---

### Exemple de document MongoDB sauvegardé en BDD

```JSON
{
  "original_name":"20230516-082245-029090.png"
  "original_path":"data/images/uploaded/20230516-082245-029090.png"
  "original_width":612
  "original_height":430
  "plotted_name":"20230516-082245-029090.png"
  "plotted_path":"data/images/plotted/20230516-082245-029090.png"
  "detection_time":"2023-05-16T08:22:45.029090"
  "speed":{
    "preprocess":16.768932342529297
    "inference":6211.602449417114
    "postprocess":3.0078887939453125
  }
  "names":{
    "0":"Fire"
    "1":"Smoke"
  }
  "boxes":{
    "0":{
      "x1":134
      "y1":174
      "x2":357
      "y2":362
      "conf":0.925
      "classname":0
    }
    "1":{
      "x1":116
      "y1":0
      "x2":450
      "y2":277
      "conf":0.9187
      "classname":1
    }
  }
}
```

---

Ce brief a pour objectif d’appliquer le **Transfer Learning** sur une base de données.

Dans ce brief, nous appliquons un apprentissage supervisé pour **détecter les incendies** à partir d’une vidéo ou image.

## Challenges

-	Labélisation des images
-	Appliquer le Transfer Learning sur YoloV5.
-	Faire des tests à partir de la webcam

## Phases de brief

### Partie 1 : Base de données

Vous devrez labéliser la DataSet fournie pour le modèle Yolo. Outil : https://www.makesense.ai

Données labelisées dans **Teams Sujet – 24 février**

### Partie 2 : Transfer Learning

Cette deuxième partie est réservée pour réaliser un Transfer Learning sur l’architecture de Yolov5.

### Partie 3 : Application

Vous êtes censés à développer une application **Streamlit** qui sera capable de :

- Charger et exécuter la détection à partir d’une image, vidéo ou d’une webcam.
- Permettre de stocker les détections dans une bdd 

##  Rendus

- Schéma de la BDDR
- Dépôt Github
- Restitution orale (au retour d’alternance)