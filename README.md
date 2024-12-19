# Projets de High-Dimensional Deep Learning

Ce dépôt contient quatre mini-projets réalisés dans le cadre du cours **High-Dimensional Deep Learning (HDDL)**. Chaque projet explore des aspects spécifiques des réseaux de neurones profonds appliqués à des tâches variées.

---

## 1. Approfondissement sur les CNNs
_Keywords_ : Convolutional layer, Pooling, Dropout, Convolutional network architectures ($\texttt{ResNet}$, $\texttt{Inception}$), Transfer learning and fine-tuning, Applications for image or signal classification, Applications for objects localization and detection.

**Objectif :** Construire et entraîner un modèle capable d'identifier l'auteur d'une peinture en utilisant le dataset "Art Challenge".  

**Étapes principales :**
- Analyse exploratoire critique de la base de données comprenant des peintures en haute et basse définition.
- Sélection et justification des architectures de réseaux adaptées.
- Entraînement et évaluation des performances.  

**Fichier associé :**
- `Notebooks/Projet 1/Projet 1.ipynb`

---

## 2. Approfondissement sur les VAEs
_Keywords_ : Encoder-decoder, Variational auto-encoder, representation learning.

**Objectif :** Implémente un CVAE (Conditional Variational Autoencoder) pour la génération d'images conditionnées sur des classes spécifiques. Le modèle est entraîné sur le dataset Fashion-MNIST, et des exemples synthétiques sont générés pour chaque classe.

**Étapes principales :**
- Entraînement du CVAE sur le dataset Fashion-MNIST.
- Génération d'images conditionnées sur des classes.
- Analyse et visualisation des résultats, y compris l'exploration de l'espace latent appris.

**Fichier associé :**
- `Notebooks/Projet 2/Projet 2.ipynb`

---

## 3. Approfondissement sur le SSL
_Keywords_ : contrastive learning, masked autoencoders.

**Objectif :** Comparer différentes stratégies d'apprentissage auto-supervisé (SSL) sur des tâches de détection d'anomalies.  

**Étapes principales :**
- Entraînement de modèles SSL (autoencodeurs masqués, modèles contrastifs, réseaux siamese...) sur des données normales.
- Évaluation des modèles sur les datasets MVTec AD (catégories : bouteille, noisette, capsule, brosse à dents) et AutoVI (catégorie : faisceau moteur).
- Analyse de la courbe ROC et calcul de l'AUROC.  

**Fichier associé :**
- `Notebooks/Projet 3/Projet 3.ipynb`

---

## 4. Approfondissement sur les RNNs
_Keywords_ : Recurrent Neural Networks, LSTM, GRU, Transformers.

**Objectif :** Comparer différentes architectures de réseaux (RNN, LSTM, GRU, MLP, CNN) pour l'analyse de sentiments sur le dataset IMDB.  

**Étapes principales :**
- Entraînement de chaque modèle avec une analyse des choix d'architectures, d'hyperparamètres et de fonctions de perte.
- Comparaison des performances en termes de temps d'entraînement, mémoire et précision.
- Ensembling des modèles via un vote majoritaire pour améliorer les résultats. 

**Fichier associé :**
- `Notebooks/Projet 4/Projet 4.ipynb`

---

## Installation
Clonez ce dépôt et installez les dépendances nécessaires via `requirements.txt` :  
```bash
git clone https://github.com/manonlcvp/Projet-HDDL.git
cd Projet-HDDL
pip install -r requirements.txt

