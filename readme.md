# 🏔️ FrozenLake: Comparaison de 4 Approches d'Apprentissage par Renforcement

Ce projet compare **4 méthodes d'apprentissage par renforcement** sur l'environnement **FrozenLake** de Gymnasium (OpenAI Gym).

## 📋 Table des Matières

- [Description](#description)
- [Méthodes Implémentées](#méthodes-implémentées)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du Code](#structure-du-code)
- [Résultats Attendus](#résultats-attendus)
- [Paramètres Configurables](#paramètres-configurables)
- [Dépendances](#dépendances)


---

## 📖 Description

**FrozenLake** est un environnement classique de Reinforcement Learning où un agent doit naviguer sur un lac gelé de 4x4 cases pour atteindre un objectif sans tomber dans les trous.

```
🟢 = Start (Départ)
❄️  = Frozen (Glace - sûr)
🕳️  = Hole (Trou - échec)
🎯 = Goal (Objectif - réussite)
```

**Caractéristiques :**
- Environnement stochastique (`is_slippery=True`) : l'agent peut glisser
- 16 états possibles (4x4 grille)
- 4 actions : ⬅️ Gauche, ⬇️ Bas, ➡️ Droite, ⬆️ Haut
- Récompense : +1 si objectif atteint, 0 sinon

---

## 🎯 Méthodes Implémentées

### 1️⃣ **Q-Learning (Tabular)**

**Principe :** Apprentissage basé sur une table Q stockant les valeurs Q(s,a) pour chaque paire état-action.

**Caractéristiques :**
- Table Q de dimension 16×4
- Mise à jour : `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]`
- Aucune approximation de fonction
- Méthode classique et simple

**Hyperparamètres :**
- Learning rate (α) : 0.1
- Discount factor (γ) : 0.95
- Epsilon decay : 0.995

---

### 2️⃣ **Deep Reinforcement Learning**

**Principe :** Réseau de neurones simple (MLP avec 1 couche cachée) qui approxime la fonction Q.

**Architecture :**
```
Input (16) → Hidden Layer (32, ReLU) → Output (4)
```

**Caractéristiques :**
- Apprentissage direct sans experience replay
- Activation ReLU
- Backpropagation manuelle
- One-hot encoding des états

**Hyperparamètres :**
- Hidden size : 32 neurones
- Learning rate : 0.01
- Discount factor (γ) : 0.95

---

### 3️⃣ **DQN (Deep Q-Network)**

**Principe :** Extension du Deep RL avec **Experience Replay** pour stabiliser l'apprentissage.

**Architecture :**
```
Input (16) → Hidden Layer (64, ReLU) → Output (4)
```

**Caractéristiques :**
- **Experience Replay Buffer** : stocke 2000 transitions
- **Batch Learning** : entraînement sur mini-batches de 32
- Meilleure stabilité que Deep RL basique
- Décorrélation des expériences

**Hyperparamètres :**
- Hidden size : 64 neurones
- Learning rate : 0.001
- Buffer size : 2000
- Batch size : 32

---

### 4️⃣ **Baseline DQN (Stable-Baselines3)**

**Principe :** Implémentation professionnelle et optimisée de DQN par Stable-Baselines3.

**Caractéristiques :**
- Architecture MLP optimisée automatiquement
- Implémentation testée et validée
- Nombreuses optimisations (target network, etc.)
- API simple et efficace

**Hyperparamètres :**
- Policy : "MlpPolicy"
- Buffer size : 10000
- Learning starts : 100
- Exploration fraction : 0.5

---

## 🛠️ Installation

### Prérequis

- Python 3.8+
- pip

### Installation des dépendances

```bash
# Cloner le repository (ou télécharger les fichiers)
git clone <votre-repo>
cd frozenlake-rl-comparison

# Créer un environnement virtuel (recommandé)
python -m venv venv

# Activer l'environnement virtuel
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Installer les dépendances
pip install gymnasium numpy matplotlib stable-baselines3
```

### Installation alternative (sans Stable-Baselines3)

Si vous ne souhaitez pas utiliser Stable-Baselines3 :

```bash
pip install gymnasium numpy matplotlib
```

Le programme détectera automatiquement l'absence de la bibliothèque et sautera la méthode 4.

---

## 🚀 Utilisation

### Exécution du programme complet

```bash
python frozen_lake_comparison.py
```

### Ce qui se passe :

1. **Entraînement** de chaque agent sur 1000 épisodes
2. **Test** de chaque agent sur 100 épisodes
3. **Affichage** des taux de succès
4. **Génération** d'un graphique comparatif (`comparison_results.png`)

### Sortie attendue

```
🏔️  FROZEN LAKE - COMPARAISON DE 4 APPROCHES
State size: 16, Action size: 4

======================================================================
1️⃣  Q-LEARNING (TABLE)
======================================================================
🟦 Q-Learning Agent: Table 16x4
...
Episode 100/1000 | Avg Reward: 0.120 | Epsilon: 0.366
...

🎯 Test: Q-Learning (100 épisodes)
✅ Taux de succès: 65.0% (65/100)

[Répété pour les 4 méthodes]

======================================================================
📊 RÉSULTATS FINAUX
======================================================================
Q-Learning          : 65.0% de succès
Deep RL             : 45.0% de succès
DQN                 : 72.0% de succès
Baseline DQN        : 78.0% de succès
```

---

## 📁 Structure du Code

```
frozen_lake_comparison.py
├── Classes
│   ├── QLearningAgent          # Q-Learning avec table
│   ├── NeuralNetwork            # Réseau de neurones custom
│   ├── DeepRLAgent              # Deep RL sans replay
│   └── DQNAgent                 # DQN avec experience replay
│
├── Fonctions Utilitaires
│   ├── state_to_vector()        # Conversion one-hot
│   ├── train_agent()            # Entraînement générique
│   ├── train_baseline_dqn()     # Entraînement Baseline
│   ├── test_agent()             # Test des agents
│   └── plot_comparison()        # Visualisation
│
└── Main
    └── Exécution séquentielle des 4 méthodes
```

---

## 📊 Résultats Attendus

### Performances Typiques (1000 épisodes)

| Méthode | Taux de Succès | Temps d'Entraînement | Stabilité |
|---------|---------------|---------------------|-----------|
| **Q-Learning** | 60-70% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Deep RL** | 40-55% | ⭐⭐⭐⭐ | ⭐⭐ |
| **DQN** | 65-75% | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Baseline DQN** | 70-80% | ⭐⭐ | ⭐⭐⭐⭐⭐ |

### Observations

- **Q-Learning** : Simple et efficace pour petits espaces d'états
- **Deep RL** : Moins stable sans experience replay
- **DQN** : Bon compromis performance/complexité
- **Baseline DQN** : Meilleure performance grâce aux optimisations

---

## ⚙️ Paramètres Configurables

### Dans le code principal (`__main__`)

```python
EPISODES = 1000  # Nombre d'épisodes d'entraînement
```

### Pour chaque agent

**Q-Learning :**
```python
learning_rate = 0.1      # Taux d'apprentissage
gamma = 0.95             # Discount factor
epsilon_decay = 0.995    # Décroissance de l'exploration
```

**Deep RL / DQN :**
```python
hidden_size = 32         # Taille de la couche cachée
learning_rate = 0.01     # Taux d'apprentissage
buffer_size = 2000       # Taille du buffer (DQN seulement)
batch_size = 32          # Taille des batches (DQN seulement)
```

**Baseline DQN :**
```python
buffer_size = 10000      # Taille du replay buffer
learning_starts = 100    # Steps avant apprentissage
exploration_fraction = 0.5  # Fraction d'exploration
```

---

## 📦 Dépendances

```txt
gymnasium>=0.29.0
numpy>=1.24.0
matplotlib>=3.7.0
stable-baselines3>=2.0.0  # Optionnel
```

### Installation complète

```bash
pip install gymnasium numpy matplotlib stable-baselines3
```

---

## 🎓 Concepts Clés

### Experience Replay
Stockage des transitions (s, a, r, s') dans un buffer pour décorréler les expériences et améliorer la stabilité.

### Epsilon-Greedy
Stratégie d'exploration : choisir une action aléatoire avec probabilité ε, sinon l'action optimale.

### One-Hot Encoding
Conversion d'un état discret en vecteur binaire pour les réseaux de neurones.

### Discount Factor (γ)
Pondération des récompenses futures : récompense totale = r + γr' + γ²r'' + ...

---

## 🐛 Troubleshooting

### Erreur : Module 'stable_baselines3' not found
```bash
pip install stable-baselines3
```

### Performances faibles
- Augmenter le nombre d'épisodes (2000-5000)
- Ajuster le learning rate
- Modifier la taille du réseau de neurones

### Pas de graphique affiché
Vérifiez que matplotlib est installé et que vous n'êtes pas en environnement headless.

---






## 📝 Notes

- FrozenLake est un environnement stochastique, les résultats peuvent varier
- Pour des résultats reproductibles, fixez le seed : `np.random.seed(42)`
- L'entraînement complet prend environ 2-5 minutes selon votre machine

---


