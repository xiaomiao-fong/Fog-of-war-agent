import os
import time
import numpy as np
import gym
import chess
import chess.engine
from gym import spaces

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks # This is useful for prediction outside of .learn()

from stable_baselines3 import PPO
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

class ChessFogEnv(gym.Env):
    """
    Custom Gym environment for fog-of-war chess using python-chess.
    White = RL agent, Black = Stockfish engine opponent.
    Observation: 13x8x8 tensor (mask + 6 White planes + 6 Black planes).
    Action: Discrete index encoding (from_square, to_square, [promotion=Q]).
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, stockfish_path, fog=False, skill_level=3):
        super(ChessFogEnv, self).__init__()
        self.action_space = spaces.Discrete(8192)
        self.observation_space = spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32)
        self.board = chess.Board()
        self._get_obs = self._get_fog_obs if fog else self._get_full_obs
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({
            "Skill Level": skill_level,
            "UCI_LimitStrength": True,
            "UCI_Elo": 1320
        })
        print(f"Engine initialized with fog {fog}")
        self.ignore_check_rules = False
        self._move_to_action_map = self._generate_move_to_action_map()
        self.old_stockfish_score = 0.0

    def _generate_move_to_action_map(self):
        move_map = {}
        for from_square in range(64):
            for to_square in range(64):
                action_idx = from_square * 64 + to_square
                move_map[chess.Move(from_square, to_square)] = action_idx
                
                promotion_action_idx = 4096 + from_square * 64 + to_square
                move_map[chess.Move(from_square, to_square, promotion=chess.QUEEN)] = promotion_action_idx
        return move_map

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        obs = self._get_obs() # Get the observation
        info = {}            # Create an empty info dictionary
        return obs, info     # Return both the observation and the info dict

    def step(self, action):
        original_board = self.board.copy()
        reward = 0.0

        move = self._action_to_move(action)

        self.board.push(move)

        def calculate_material_score(board, color):
            score = 0
            for piece_type in chess.PIECE_TYPES:
                for square in board.pieces(piece_type, color):
                    if piece_type == chess.PAWN: score += 1
                    elif piece_type == chess.KNIGHT or piece_type == chess.BISHOP: score += 3
                    elif piece_type == chess.ROOK: score += 5
                    elif piece_type == chess.QUEEN: score += 9
            return score

        def stockfish_board_score(board):
            score = self.engine.analyse(board, chess.engine.Limit(time=0.1))["score"].white().score()
            if score is None:
                score = -600
            reward = score - self.old_stockfish_score
            self.old_stockfish_score = score
            return reward
            
        current_white_material = calculate_material_score(self.board, chess.WHITE)
        current_black_material = calculate_material_score(self.board, chess.BLACK)
        original_white_material = calculate_material_score(original_board, chess.WHITE)
        original_black_material = calculate_material_score(original_board, chess.BLACK)

        reward += (current_white_material - original_white_material)
        reward -= (current_black_material - original_black_material)

        if self.board.is_check():
            reward += 0.1

        terminated = False # Renamed 'done' to 'terminated' for clarity with new API
        truncated = False  # Added 'truncated' for new Gym API
        info = {}

        if self.board.king(chess.BLACK) is None: # White captured Black's king
             terminated = True
             reward += 50.0 # Large reward for capturing king
             info['outcome'] = 'white_captured_king'
        elif self.board.king(chess.WHITE) is None: # Black captured White's king
             terminated = True
             reward -= 50.0 # Large penalty for losing king
             info['outcome'] = 'black_captured_king'
        elif self.board.is_game_over(): # Other standard game over conditions (checkmate, stalemate, etc.)
            terminated = True
            result = self.board.result()
            if result == '1-0':
                reward += 30.0
            elif result == '1/2-1/2':
                reward += 0.0
            else: # 0-1
                reward -= 30.0
            info['outcome'] = result

        if not terminated and not truncated: # Only play Stockfish if game is not over
            reward += stockfish_board_score(self.board) * 0.1
            result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
            self.board.push(result.move)

            post_stockfish_white_material = calculate_material_score(self.board, chess.WHITE)
            reward -= (original_white_material - post_stockfish_white_material)

            if self.board.is_game_over():
                terminated = True
                result = self.board.result()
                if result == '0-1':
                    reward -= 30.0
                elif result == '1/2-1/2':
                    reward += 0.0
                else:
                    reward += 30.0
            
        reward += max(-1.0, 0.5 - self.board.fullmove_number * 0.01) # encourage early surviving

        return self._get_obs(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)

        if self.ignore_check_rules:
            # Iterate over pseudo-legal moves instead of legal_moves
            for move in self.board.pseudo_legal_moves:
                if move in self._move_to_action_map:
                    action_idx = self._move_to_action_map[move]
                    mask[action_idx] = True
        else:
            # Original logic for standard chess
            for move in self.board.legal_moves:
                if move in self._move_to_action_map:
                    action_idx = self._move_to_action_map[move]
                    mask[action_idx] = True
        return mask

    def _action_to_move(self, action):
        if action < 4096:
            from_square = action // 64
            to_square = action % 64
            return chess.Move(from_square, to_square)
        elif action < 8192:
            action -= 4096
            from_square = action // 64
            to_square = action % 64
            return chess.Move(from_square, to_square, promotion=chess.QUEEN)
        else:
            raise ValueError(f"Invalid action index: {action}")

    def _get_fog_obs(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        white_planes = np.zeros((6, 8, 8), dtype=np.uint8)
        black_planes = np.zeros((6, 8, 8), dtype=np.uint8)
        for pt in range(1, 7):
            for sq in self.board.pieces(pt, chess.WHITE):
                mask[sq // 8, sq % 8] = 1
                white_planes[pt - 1, sq // 8, sq % 8] = 1
        for move in self.board.legal_moves:
            if self.board.color_at(move.from_square) == chess.WHITE:
                to_sq = move.to_square
                mask[to_sq // 8, to_sq % 8] = 1
                if self.board.is_capture(move):
                    captured = self.board.piece_at(to_sq)
                    if captured:
                        black_planes[captured.piece_type - 1, to_sq // 8, to_sq % 8] = 1
                piece = self.board.piece_at(move.from_square)
                if piece:
                    if piece.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                        fx, fy = move.from_square % 8, move.from_square // 8
                        tx, ty = to_sq % 8, to_sq // 8
                        dx = np.sign(tx - fx)
                        dy = np.sign(ty - fy)
                        cx, cy = fx + dx, fy + dy
                        while (cx, cy) != (tx, ty):
                            mask[cy, cx] = 1
                            cx += dx; cy += dy
                    if piece.piece_type == chess.PAWN and abs(move.to_square - move.from_square) == 16:
                        mid = (move.from_square + move.to_square) // 2
                        mask[mid // 8, mid % 8] = 1
        obs = np.vstack((mask[None, ...], white_planes, black_planes)).astype(np.float32)
        return obs

    def _get_full_obs(self):
        mask = np.ones((8, 8), dtype=np.uint8)
        white_planes = np.zeros((6, 8, 8), dtype=np.uint8)
        black_planes = np.zeros((6, 8, 8), dtype=np.uint8)

        for pt in range(1, 7):
            for sq in self.board.pieces(pt, chess.WHITE):
                white_planes[pt - 1, sq // 8, sq % 8] = 1
            for sq in self.board.pieces(pt, chess.BLACK):
                black_planes[pt - 1, sq // 8, sq % 8] = 1

        obs = np.vstack((mask[None, ...], white_planes, black_planes)).astype(np.float32)
        return obs

    def render(self, mode='human'):
        print(self.board)

    def close(self):
        if self.engine:
            self.engine.quit()
        self.engine = None


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for the ChessFogEnv.
    Input observation space: (13, 8, 8)
    Output features: will be fed to the actor and critic networks.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume the input image is (n_channels, height, width)
        # where n_channels is 13 (mask + 6 white + 6 black planes)
        # and height, width are 8x8.
        n_input_channels = observation_space.shape[0]

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

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# === Training Code ===
if __name__ == "__main__":
    STOCKFISH_PATH = YOUR_PATH_HERE
    env = ChessFogEnv(STOCKFISH_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )

    MODEL_PATH = "chess_ppo_model.zip"
    MODEL_LOADED = True

    if MODEL_LOADED:
        if os.path.exists(MODEL_PATH):
            print(f"Loading existing masked model from {MODEL_PATH} for further training...")
            model = MaskablePPO.load(
                MODEL_PATH,
                env=env,
                device=device,
                # policy_kwargs=policy_kwargs,
                custom_objects={'features_extractor_class': CustomCNN},
                print_system_info=True
            )
        else:
            print("No existing masked model found. Starting fresh training...")
            model = MaskablePPO(
                MaskableActorCriticPolicy,
                env,
                verbose=1,
                device=device,
                policy_kwargs=policy_kwargs,
                tensorboard_log="./logs/ppo_chess_fog_masked"
            )
    else:
        print("Starting fresh training...")
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            verbose=1,
            device=device,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./logs/ppo_chess_fog_masked"
        )

    try:
        print("Continuing training...")
        model.learn(total_timesteps=500000, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Training stopped.")
    finally:
        model.save(MODEL_PATH)
        model.save(f'{MODEL_PATH}_{int(time.time())}')
        env.close()
        print(f"Model saved to {MODEL_PATH}, {MODEL_PATH}_{int(time.time())}")

