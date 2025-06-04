# Intro to AI 2025 Spring final project - Fog of war Agent
This is an AI chess engine made for a variant of chess named Fog of War chess.

---
Group Members : 林子鈞、葉羽宸、潘仰祐、林宥廷

## Overview
### Fog of War Chess
[Fog of War Chess]( https://en.wikipedia.org/wiki/Dark_chess ) (aka. Dark Chess) is a variant of chess. It is named so due to its characteristic of incomplete information - your chess piece types are invisible to your opponent; they can only see their own chess pieces and positions they can legally move to, and vice versa.

### Prerequisite
Please refer to [`requirements.txt`]( ./requirements.txt ). Run `pip3 install -r requirements.txt` to install them all. The working versions of them are:
```
python==3.10.9           # 3.11.2 won't work
python-chess==1.11.2
stable-baselines3==2.6.0
sb3_contrib==2.6.0
gym==0.26.0
numpy==1.26.4
torch==2.4.1             # +cu121
pygame==2.6.1
```

The coding environment is mainly on `Windows-10-10.0.22631-SP0`. But Ubuntu/Debian with kernel `Linux-6.1.0` (e.g. Debian 12) should also work.

### Usage
Use the following command to play against our agent:
``python GUIgame.py``

### Hyperparameter
CNN Model:
```
self.cnn = nn.Sequential(
    nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Flatten(),
)

# Compute shape by doing one forward pass
with torch.no_grad():
    n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
```
Actor-Critic Model:
```
Actor: linear, 256 -> 64 -> 64 -> 8192, full dense
Critic: linear, 256 -> 64 -> 64 -> 1, full dense
```

Algorithm:
```
MaskablePPO Policy: sb3_contrib.common.maskable.policies.MaskableActorCriticPolicy
### sb3 defaults ###
learning rate: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.0
vf_coef: 0.5
max_grad_norm: 0.5
####################
```

### Experiment results

The results show that our greedy agent performs great against RandomAgent, but our trained agents don't.
This indicates that our agents didn't learn good policies. This might be due to:
1. The absence of belief state estimation over our fogged states.
2. Not enough training steps.
3. Our reward mechanism may make our agent overly defensive.

![image](https://github.com/user-attachments/assets/e8a29f55-3b55-4485-bbe1-89524471d659)
![image](https://github.com/user-attachments/assets/48c86fb1-5543-40f4-9234-8bad9c49d77a)
![image](https://github.com/user-attachments/assets/7fadca5c-5f1d-4d7b-9984-3b2dec1168c8)

Another interesting observation is that black has an advantage over white when played in greedy or random policy.
This might suggest that early exposure poses a weakness to your opponent.<br>
![image](https://github.com/user-attachments/assets/e705bcca-4f18-44a6-baca-a0e1378c82c2)
![image](https://github.com/user-attachments/assets/cf30b9b5-86ef-4bc8-8404-8eb725cd26e2)


