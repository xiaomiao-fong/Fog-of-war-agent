import chess
import chess.engine
import numpy as np
import torch
import gym
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

from agents import BaseAgent # Assuming BaseAgent.py is in the same directory

# Re-define ChessFogEnv and CustomCNN as they are necessary for loading the model
# and for the agent to understand the observation space and action space.
# We'll adapt ChessFogEnv slightly for agent usage, primarily for _action_to_move
# and potentially a simplified _get_obs if not directly using the full env.

class ChessFogEnvForAgent(gym.Env):
    """
    A minimal ChessFogEnv necessary for loading the MaskablePPO model
    and for the agent to understand the action space and observation space.
    It does not require the Stockfish engine for action selection,
    as the agent will decide the move.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, fog=True):
        super(ChessFogEnvForAgent, self).__init__()
        self.action_space = spaces.Discrete(8192)
        self.observation_space = spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32)
        self.board = chess.Board() # Board is needed for action masks and observation
        self._get_obs = self._get_fog_obs if fog else self._get_full_obs
        self._move_to_action_map = self._generate_move_to_action_map()

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
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        raise NotImplementedError("Step method not implemented for agent-side environment.")

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
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
            for sq in self.board.pieces(pt, chess.BLACK):
                mask[sq // 8, sq % 8] = 1
                black_planes[pt - 1, sq // 8, sq % 8] = 1
        for move in self.board.legal_moves:
            if self.board.color_at(move.from_square) == chess.BLACK:
                to_sq = move.to_square
                mask[to_sq // 8, to_sq % 8] = 1
                if self.board.is_capture(move):
                    captured = self.board.piece_at(to_sq)
                    if captured:
                        white_planes[captured.piece_type - 1, to_sq // 8, to_sq % 8] = 1
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
        obs = np.vstack((mask[None, ...], black_planes, white_planes)).astype(np.float32)
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
        pass # No engine to close for this minimal env

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for the ChessFogEnv.
    Input observation space: (13, 8, 8)
    Output features: will be fed to the actor and critic networks.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
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

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class RLAgent(BaseAgent):
    def __init__(self, name="ChessFogRLAgent", model_path="models/CNN_RL/chess_fog_ppo_model.zip", fog=True):
        super().__init__(name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = ChessFogEnvForAgent(fog=fog)
        self.model = self._load_model(model_path)
        self.fog = fog

    def _load_model(self, model_path):
        if os.path.exists(model_path):
            print(f"Loading trained model from {model_path}...")
            model = MaskablePPO.load(
                model_path,
                env=self.env,
                device=self.device,
                custom_objects={'features_extractor_class': CustomCNN},
                print_system_info=True
            )
            print("Model loaded successfully.")
            return model
        else:
            raise FileNotFoundError(f"Trained model not found at {model_path}. Please ensure it exists.")

    def act(self, board: chess.Board) -> chess.Move:
        """
        Selects a move based on the loaded MaskablePPO model.
        """
        self.env.board = board.copy()

        obs = self.env._get_obs()
        action_masks = self.env.action_masks()

        obs_tensor = torch.as_tensor(obs[None]).float()
        action_masks_tensor = torch.as_tensor(action_masks[None])

        obs_tensor_cpu = obs_tensor.cpu()
        action_masks_tensor_cpu = action_masks_tensor.cpu()

        action, _states = self.model.predict(
            obs_tensor_cpu, # Pass the CPU tensor
            action_masks=action_masks_tensor_cpu, # Pass the CPU tensor
            deterministic=True
        )

        predicted_action_idx = action.item()

        move = self.env._action_to_move(predicted_action_idx)

        if move in board.pseudo_legal_moves:
            return move
        else:
            print(f"Warning: Model predicted an illegal move: {move}. Falling back to a random legal move.")
            return list(board.pseudo_legal_moves)[0] if list(board.pseudo_legal_moves) else None


if __name__ == "__main__":
    print("This is only for import")
    # # Example usage:
    # # Make sure 'chess_fog_ppo_model.zip' is in the same directory
    # try:
    #     agent = ChessFogAgent(name="MyRLChessAgent", model_path="chess_fog_ppo_model.zip", fog=True)
        
    #     # Simulate a game turn
    #     current_board = chess.Board()
    #     print("Initial Board:")
    #     print(current_board)

    #     # Agent makes a move (White's turn)
    #     agent_move = agent.act(current_board)
    #     if agent_move:
    #         print(f"Agent chose move: {agent_move}")
    #         current_board.push(agent_move)
    #         print("\nBoard after agent's move:")
    #         print(current_board)
    #     else:
    #         print("Agent could not find a legal move.")

    #     # Simulate Black's turn (e.g., Stockfish or random move)
    #     # For demonstration, let's just make a random move if it's not game over
    #     if not current_board.is_game_over():
    #         legal_moves_black = list(current_board.legal_moves)
    #         if legal_moves_black:
    #             black_move = legal_moves_black[0] # Just take the first legal move for simplicity
    #             print(f"Black (opponent) chose move: {black_move}")
    #             current_board.push(black_move)
    #             print("\nBoard after opponent's move:")
    #             print(current_board)
    #         else:
    #             print("No legal moves for Black.")

    #     print(f"\nGame Over: {current_board.is_game_over()}, Result: {current_board.result()}")


    # except FileNotFoundError as e:
    #     print(e)
    # except Exception as e:
    #     print(f"An error occurred: {e}")