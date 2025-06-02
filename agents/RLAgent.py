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

from .BaseAgent import BaseAgent



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
        for move in self.board.pseudo_legal_moves:
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
        for move in self.board.pseudo_legal_moves:
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
    def __init__(self, model_path="models/CNN_RL/chess_fog_ppo_model.zip", fog=True):
        super().__init__("ChessFogRLAgent")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = ChessFogEnvForAgent(fog=fog)
        self.model = self._load_model(model_path)
        self.fog = fog
        self._flipped_square_map = {i: chess.square_mirror(i) for i in range(64)}

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
        original_board = board.copy()

        if original_board.turn == chess.BLACK:
            flipped_board = original_board.mirror()
            flipped_board.turn = chess.WHITE 
        else:
            flipped_board = original_board.copy()
            
        self.env.board = flipped_board 

        obs = self.env._get_obs()
        action_masks = self.env.action_masks()

        obs_tensor = torch.as_tensor(obs[None]).float()
        action_masks_tensor = torch.as_tensor(action_masks[None])

        action, _states = self.model.predict(
            obs_tensor,
            action_masks=action_masks_tensor,
            deterministic=True
        )

        predicted_action_idx = action.item()
        predicted_move_on_flipped_board = self.env._action_to_move(predicted_action_idx)

        # If the board was flipped, flip the predicted move back to the original board's perspective
        if original_board.turn == chess.BLACK:
            # Flip the from_square and to_square of the predicted move
            from_square_orig = self._flipped_square_map[predicted_move_on_flipped_board.from_square]
            to_square_orig = self._flipped_square_map[predicted_move_on_flipped_board.to_square]
            
            # Create the final move for the original board
            final_move = chess.Move(from_square_orig, to_square_orig, 
                                    promotion=predicted_move_on_flipped_board.promotion)
        else:
            final_move = predicted_move_on_flipped_board

        if final_move in board.pseudo_legal_moves:
            return final_move
        else:
            print(f"Warning: Model predicted an illegal move: {final_move}. Falling back to a random legal move.")
            return list(board.pseudo_legal_moves)[0] if list(board.pseudo_legal_moves) else None


if __name__ == "__main__":
    print("This is only for import")