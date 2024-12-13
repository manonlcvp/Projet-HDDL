# data manipulation
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# text processing
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
stopwords = set(stopwords.words('english'))

# pytorch
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

# sklearn
from sklearn.metrics import classification_report, confusion_matrix

# utils
import os
from tqdm import tqdm
tqdm.pandas()
from collections import Counter
import optuna
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_pred = pd.read_csv('data/IMDB_Pred.csv')

texts = data_pred.processed.values 
words = ' '.join(texts) # fusionne tous les textes en un seul

words = words.split()

# Création des dictionnaires
counter = Counter(words)
vocab = sorted(counter, key=counter.get, reverse=True) # trie les mots par fréquence décroissante
int2word = dict(enumerate(vocab, 1))
int2word[0] = '<PAD>'
word2int = {word: id for id, word in int2word.items()}

texts_enc = [[word2int[word] for word in review.split()] for review in tqdm(texts)]

def pad_features(texts, pad_id, seq_length=256):
    # pad_id est l'indice de l'identifiant spécial <PAD> 

    sequences = np.full((len(texts), seq_length), pad_id, dtype=int) # initialisation d'une matrice feature de longueur seq_length avec l'element <PAD>

    for i, row in enumerate(texts):
        sequences[i, :len(row)] = np.array(row)[:seq_length] # feature est completee par les elements de la sequence d'entree review

    return sequences

seq_length = 256
sequences = pad_features(texts_enc, pad_id=word2int['<PAD>'], seq_length=seq_length)

assert len(sequences) == len(texts_enc)
assert len(sequences[0]) == seq_length

sequences[:10, :3] # 3 premiers élèments des 10 premières séquences

labels = data_pred.label.to_numpy()

# Choix des ratios pour la séparation des données
train_size = .8     # utilisation de 80% des données pour l'apprentissage et 20% pour le test
val_size = .5       # utilisation de 50% des données test pour la validation (Total : 80% apprentissage, 10% validation, 10% test)

# Données d'apprentissage
split_id = int(len(sequences) * train_size)
train_x, remain_x = sequences[:split_id], sequences[split_id:]
train_y, remain_y = labels[:split_id], labels[split_id:]

# Données de validation et de test
split_val_id = int(len(remain_x) * val_size)
val_x, test_x = remain_x[:split_val_id], remain_x[split_val_id:]
val_y, test_y = remain_y[:split_val_id], remain_y[split_val_id:]

batch_size = 128

# Création des tenseurs
train_set = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_set = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_set = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# Création des DataLoaders
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(valid_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

vocab_size = len(word2int)
output_size = 1
embedding_size = 256
hidden_size = 512
n_layers = 2
dropout=0.25
dropout_layer = 0.3
lr = 0.001
epochs = 5

grad_clip = 5 # paramètre pour l'apprentisage

class Modele_LSTM(nn.Module):
    def __init__(self, vocab_size, output_size, hidden_size=128, embedding_size=400, n_layers=2, dropout=0.2, dropout_layer = 0.3):
        super(Modele_LSTM, self).__init__()

        # embedding layer is useful to map input into vector representation
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout_layer) # desactivation de 30% des neurones (pour eviter le surapprentissage) / valeur differente du dropout des paramètres
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid() # fonction d'activation de sortie = sigmoïde (classification binaire)

    def forward(self, x):
        
        x = x.long() # conversion en entiers longs
        x = self.embedding(x)
        o, _ =  self.lstm(x) # 'o' = la sortie à chaque pas de temps
        o = o[:, -1, :] # Récupération de la dernière sortie temporelle
        o = self.dropout(o) # Désactivation aléatoire des neurones
        o = self.fc(o) # couche entièrement connectée pour la sortie, output_size=1 car classification binaire
        o = self.sigmoid(o) # fonction d'activation sigmoïde pour la prédiction (renvoie une proba entre 0 et 1 d'appartenir à la classe positive)

        return o

criterion = nn.BCELoss()  # Binary Cross-Entropy car classification binaire
print_every = 1
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'epochs': epochs
}
es_limit = 5

def optim_LSTM(trial):
    # Définir les hyperparamètres à tester
    embedding_size = trial.suggest_int('embedding_size', 128, 512)
    hidden_size = trial.suggest_int('hidden_size', 128, 512)
    dropout = trial.suggest_uniform('dropout', 0.2, 0.5)
    dropout_layer = trial.suggest_uniform('dropout_layer', 0.2, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-3)
    batch_size = trial.suggest_int('batch_size', 64, 128)
    epochs = trial.suggest_int('epochs', 5, 15)
    
    # Créer un modèle avec ces hyperparamètres
    model = Modele_LSTM(vocab_size, output_size, 
                        embedding_size=embedding_size, 
                        hidden_size=hidden_size, 
                        dropout=dropout,
                        dropout_layer=dropout_layer).to(device)
    
    # Définir l'optimiseur et la fonction de perte
    optimizer = Adam(model.parameters(), lr=lr)
    
    # DataLoader pour charger les données par batch
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    
    # Affichage de la barre de progression pour les époques
    epochloop = tqdm(range(epochs), position=0, desc=f"Trial {trial.number} | Training", leave=True)
    
    # Entraîner le modèle pendant les 'epochs' suggérées
    for epoch in epochloop:
        model.train()
        train_loss = 0
        for feature, target in train_loader:
            feature, target = feature.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(feature)
            loss = criterion(out.squeeze(), target.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation du modèle
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for feature, target in val_loader:
                feature, target = feature.to(device), target.to(device)
                out = model(feature)
                loss = criterion(out.squeeze(), target.float())
                val_loss += loss.item()

        # Mise à jour de la barre de progression de l'époch avec la validation
        epochloop.set_postfix({'val_loss': val_loss / len(val_loader)})
    
    # Retourner la perte de validation, qu'Optuna va minimiser
    return val_loss / len(val_loader)

# Créer un objet study Optuna
opti_LSTM = optuna.create_study(direction='minimize')  # On minimise la perte

# Lancer l'optimisation
opti_LSTM.optimize(optim_LSTM, n_trials=25)  # Teste 25 configurations différentes

# Enregistrer les meilleurs hyperparamètres trouvés
best_params = opti_LSTM.best_params

# Chemin du fichier où les hyperparamètres seront enregistrés
json_file_path = 'best_hyperparameters.json'

# Sauvegarder sous forme de fichier JSON
with open(json_file_path, 'w') as json_file:
    json.dump(best_params, json_file)
