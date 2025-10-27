import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random
import os

# Pour Stable-Baselines3 (installer avec: pip install stable-baselines3)
try:
    from stable_baselines3 import DQN as BaselineDQN
    from stable_baselines3.common.evaluation import evaluate_policy
    BASELINE_AVAILABLE = True
except ImportError:
    BASELINE_AVAILABLE = False
    print("⚠️  Stable-Baselines3 non installé. Installez avec: pip install stable-baselines3")


# ============================================================================
# 1. Q-LEARNING (TABLE)
# ============================================================================
class QLearningAgent:
    """Agent utilisant Q-Learning classique avec table Q"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        print(f"🟦 Q-Learning Agent: Table {state_size}x{action_size}")
    
    def act(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """Mise à jour Q-Learning"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ============================================================================
# 2. DEEP REINFORCEMENT LEARNING (Réseau de neurones simple)
# ============================================================================
class NeuralNetwork:
    """Réseau de neurones simple avec une couche cachée"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        # Initialisation Xavier
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        
        print(f"🟩 Neural Network: {input_size} → {hidden_size} → {output_size}")
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        """Forward pass"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        return self.z2
    
    def backward(self, x, y_true):
        """Backward pass"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(1, -1)
        
        m = x.shape[0]
        
        # Forward
        y_pred = self.forward(x)
        
        # Backward
        dz2 = (y_pred - y_true) / m
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dw1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
    
    def predict(self, x):
        return self.forward(x)


class DeepRLAgent:
    """Agent Deep RL sans experience replay"""
    
    def __init__(self, state_size, action_size, hidden_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        
        self.model = NeuralNetwork(state_size, hidden_size, action_size, self.learning_rate)
    
    def act(self, state):
        """Epsilon-greedy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def learn(self, state, action, reward, next_state, done):
        """Apprentissage direct (sans replay)"""
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
        
        target_f = self.model.predict(state)
        target_f[0][action] = target
        
        self.model.backward(state, target_f)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ============================================================================
# 3. DQN (Deep Q-Network avec Experience Replay)
# ============================================================================
class DQNAgent:
    """Agent DQN complet avec experience replay"""
    
    def __init__(self, state_size, action_size, hidden_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.model = NeuralNetwork(state_size, hidden_size, action_size, self.learning_rate)
        
        print(f"🟪 DQN Agent with Experience Replay (buffer size: 2000)")
    
    def remember(self, state, action, reward, next_state, done):
        """Stocker dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Epsilon-greedy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """Experience replay"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            self.model.backward(state, target_f)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ============================================================================
# UTILITAIRES
# ============================================================================
def state_to_vector(state, state_size):
    """Convertir état entier en one-hot vector"""
    vector = np.zeros(state_size)
    vector[state] = 1.0
    return vector


def train_agent(agent, env, agent_name, episodes=500, use_vector=False):
    """Fonction générique d'entraînement"""
    state_size = env.observation_space.n
    rewards_history = []
    success_rate = []
    
    print(f"\n{'='*70}")
    print(f"🚀 Entraînement: {agent_name}")
    print(f"{'='*70}")
    
    for episode in range(episodes):
        state, _ = env.reset()
        if use_vector:
            state_vec = state_to_vector(state, state_size)
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            # Action
            if use_vector:
                action = agent.act(state_vec)
            else:
                action = agent.act(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if use_vector:
                next_state_vec = state_to_vector(next_state, state_size)
            
            # Apprentissage
            if hasattr(agent, 'remember'):
                # DQN avec replay
                if use_vector:
                    agent.remember(state_vec, action, reward, next_state_vec, done)
                else:
                    agent.remember(state, action, reward, next_state, done)
            else:
                # Q-Learning ou Deep RL
                if use_vector:
                    agent.learn(state_vec, action, reward, next_state_vec, done)
                else:
                    agent.learn(state, action, reward, next_state, done)
            
            if use_vector:
                state_vec = next_state_vec
            else:
                state = next_state
            
            total_reward += reward
            steps += 1
        
        # Replay pour DQN
        if hasattr(agent, 'replay'):
            agent.replay(batch_size=32)
        
        rewards_history.append(total_reward)
        
        # Taux de succès
        if len(rewards_history) >= 100:
            success = np.mean(rewards_history[-100:])
            success_rate.append(success)
        
        # Affichage
        if (episode + 1) % 100 == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"Episode {episode+1}/{episodes} | Avg Reward: {avg:.3f} | Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history, success_rate


def train_baseline_dqn(env, episodes=10000):
    """Entraîner avec Stable-Baselines3 DQN"""
    if not BASELINE_AVAILABLE:
        print("❌ Stable-Baselines3 non disponible")
        return None, []
    
    print(f"\n{'='*70}")
    print(f"🚀 Entraînement: Baseline DQN (Stable-Baselines3)")
    print(f"{'='*70}")
    
    model = BaselineDQN(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        buffer_size=10000,
        learning_starts=100,
        batch_size=32,
        gamma=0.95,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        verbose=0
    )
    
    # Entraînement avec callback pour tracking
    rewards_history = []
    
    for i in range(episodes // 100):
        model.learn(total_timesteps=100, reset_num_timesteps=False)
        
        # Évaluation
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        rewards_history.append(mean_reward)
        
        if (i + 1) % 10 == 0:
            print(f"Steps {(i+1)*100}/{episodes} | Avg Reward: {mean_reward:.3f}")
    
    return model, rewards_history


def test_agent(agent, env, agent_name, episodes=100, use_vector=False, is_baseline=False):
    """Tester l'agent"""
    state_size = env.observation_space.n
    total_rewards = []
    
    print(f"\n🎯 Test: {agent_name} ({episodes} épisodes)")
    
    for episode in range(episodes):
        state, _ = env.reset()
        if use_vector and not is_baseline:
            state_vec = state_to_vector(state, state_size)
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            if is_baseline:
                action, _ = agent.predict(state, deterministic=True)
                action = int(action)  # CORRECTION: Convertir en entier
            elif use_vector:
                q_values = agent.model.predict(state_vec)
                action = np.argmax(q_values[0])
            else:
                if hasattr(agent, 'model'):
                    q_values = agent.model.predict(state_to_vector(state, state_size))
                    action = np.argmax(q_values[0])
                else:
                    action = np.argmax(agent.q_table[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if use_vector and not is_baseline:
                state_vec = state_to_vector(next_state, state_size)
            else:
                state = next_state
            
            total_reward += reward
            steps += 1
        
        total_rewards.append(total_reward)
    
    success_rate = np.mean(total_rewards)
    print(f"✅ Taux de succès: {success_rate*100:.1f}% ({int(success_rate*episodes)}/{episodes})")
    
    return success_rate


def plot_comparison(results, save_path='comparison_results.png'):
    """Graphique comparatif"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'green', 'purple', 'red']
    
    for idx, (name, rewards) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        
        # Courbe des récompenses
        ax.plot(rewards, alpha=0.3, color=colors[idx], label='Récompense')
        
        # Moyenne mobile
        if len(rewards) >= 50:
            window = 50
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), smoothed, 
                   color=colors[idx], linewidth=2, label='Moyenne (50)')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Récompense')
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphique sauvegardé: {save_path}")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Environnement
    env = gym.make("FrozenLake-v1", is_slippery=True)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    print("🏔️  FROZEN LAKE - COMPARAISON DE 4 APPROCHES")
    print(f"State size: {state_size}, Action size: {action_size}\n")
    
    EPISODES = 1000
    results = {}
    test_results = {}
    
    # 1. Q-LEARNING
    print("\n" + "="*70)
    print("1️⃣  Q-LEARNING (TABLE)")
    print("="*70)
    agent_q = QLearningAgent(state_size, action_size)
    rewards_q, _ = train_agent(agent_q, env, "Q-Learning", episodes=EPISODES, use_vector=False)
    results["1. Q-Learning"] = rewards_q
    test_results["Q-Learning"] = test_agent(agent_q, env, "Q-Learning", episodes=100, use_vector=False)
    
    # 2. DEEP RL (sans replay)
    print("\n" + "="*70)
    print("2️⃣  DEEP REINFORCEMENT LEARNING (Neural Network)")
    print("="*70)
    agent_deep = DeepRLAgent(state_size, action_size, hidden_size=32)
    rewards_deep, _ = train_agent(agent_deep, env, "Deep RL", episodes=EPISODES, use_vector=True)
    results["2. Deep RL"] = rewards_deep
    test_results["Deep RL"] = test_agent(agent_deep, env, "Deep RL", episodes=100, use_vector=True)
    
    # 3. DQN (avec replay)
    print("\n" + "="*70)
    print("3️⃣  DQN (Deep Q-Network avec Experience Replay)")
    print("="*70)
    agent_dqn = DQNAgent(state_size, action_size, hidden_size=64)
    rewards_dqn, _ = train_agent(agent_dqn, env, "DQN", episodes=EPISODES, use_vector=True)
    results["3. DQN"] = rewards_dqn
    test_results["DQN"] = test_agent(agent_dqn, env, "DQN", episodes=100, use_vector=True)
    
    # 4. BASELINE DQN (Stable-Baselines3)
    if BASELINE_AVAILABLE:
        print("\n" + "="*70)
        print("4️⃣  BASELINE DQN (Stable-Baselines3)")
        print("="*70)
        agent_baseline, rewards_baseline = train_baseline_dqn(env, episodes=EPISODES)
        if agent_baseline:
            # Adapter pour le graphique (répéter les valeurs)
            rewards_baseline_full = []
            for r in rewards_baseline:
                rewards_baseline_full.extend([r] * 100)
            results["4. Baseline DQN"] = rewards_baseline_full
            test_results["Baseline DQN"] = test_agent(agent_baseline, env, "Baseline DQN", 
                                                      episodes=100, use_vector=False, is_baseline=True)
    
    # RÉSULTATS FINAUX
    print("\n" + "="*70)
    print("📊 RÉSULTATS FINAUX")
    print("="*70)
    for name, score in test_results.items():
        print(f"{name:20s}: {score*100:.1f}% de succès")
    
    # Graphique comparatif
    plot_comparison(results)
    
    print("\n✨ Comparaison terminée avec succès!")