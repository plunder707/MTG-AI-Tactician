import pandas as pd
import random
import logging
import re
import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pygame
from pygame.locals import QUIT
from typing import List, Optional, Dict, Any
from transitions import Machine
from torch.utils.data import DataLoader, Dataset
import cv2
import base64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SETUP LOGGING ---
def setup_logger(name: str, log_file: str, level=logging.INFO):
    """
    Sets up a logger with the specified name and log file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger('mtg_logger', 'mtg_game.log')

class QNetwork(nn.Module):
    def __init__(self, input_dim, action_size, latent_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.attention_aggregation = AttentionAggregation(latent_dim)
        self.fc_out = nn.Linear(latent_dim, action_size)

    def forward(self, self_features, other_features):
        # Process self features
        x = torch.relu(self.fc1(self_features))  # Shape: (batch_size, latent_dim)
        # Process other features
        other_x = torch.relu(self.fc1(other_features))  # Shape: (batch_size, latent_dim)
        # Adjust dimensions for attention layers
        x = x.unsqueeze(1)  # Shape: (batch_size, seq_len=1, latent_dim)
        other_x = other_x.unsqueeze(1)  # Shape: (batch_size, seq_len=1, latent_dim)
        # Aggregate features using attention
        attn_output = self.attention_aggregation(x, other_x)  # Shape: (batch_size, seq_len=1, latent_dim)
        # Flatten and pass through the output layer
        attn_output = attn_output.squeeze(1)  # Shape: (batch_size, latent_dim)
        output = self.fc_out(attn_output)
        return output


# --- SYMBOLIC ENGINE: For Card Synergies & Strategies ---
class SymbolicMTGEngine:
    """
    Uses symbolic logic to compute optimal card plays, synergies, and strategies.
    Dynamically generates formulas based on the current game state.
    """
    def __init__(self):
        # Define initial symbolic variables for card effects, mana, and game state
        self.a1, self.an, self.d, self.n, self.Sn = sp.symbols('a1 an d n Sn')
        self.formulas = [
            sp.Eq(self.an, self.a1 + (self.n - 1) * self.d),
            sp.Eq(self.Sn, (self.n / 2) * (2 * self.a1 + (self.n - 1) * self.d)),
        ]

    def generate_dynamic_formulas(self, game_state: 'GameState') -> List[sp.Eq]:
        """
        Generates dynamic formulas based on the current game state.
        Game state is used to adapt strategies and evaluate card synergies.
        """
        # Extract relevant game state information
        available_mana = game_state.current_player.mana_pool
        creatures_on_battlefield = len([card for card in game_state.current_player.battlefield if card.is_creature()])
        opponent_creatures = len([card for card in game_state.opponent.battlefield if card.is_creature()])
        player_health = game_state.current_player.health
        opponent_health = game_state.opponent.health

        # Define symbolic variables for current game conditions
        mana, creatures, opponent_creatures, health, opponent_health = sp.symbols('mana creatures opp_creatures health opp_health')

        # Dynamically create formulas based on game state
        formulas = []

        # Formula 1: Maximize health difference
        formulas.append(sp.Eq(self.Sn, health - opponent_health))

        # Formula 2: Mana efficiency (cards with lower mana cost but high impact are prioritized)
        if available_mana:
            avg_mana_cost = sum(available_mana.values()) / len(available_mana)
            formulas.append(sp.Eq(mana, avg_mana_cost))
        
        # Formula 3: Board presence (maximize creature advantage)
        if creatures_on_battlefield > 0:
            formulas.append(sp.Eq(creatures, creatures_on_battlefield - opponent_creatures))

        # Formula 4: Strategic threshold for life total (adjust based on health)
        if player_health < 10:
            # If health is low, prioritize defensive strategies
            formulas.append(sp.Eq(self.a1, player_health * 1.5))
        else:
            # Otherwise, focus on aggressive plays
            formulas.append(sp.Eq(self.a1, (player_health - opponent_health) * 1.2))

        # Additional dynamic formulas can be added based on game state conditions
        return formulas

    def compute_synergy(self, cards: List['Card'], game_state: 'GameState') -> Optional['Card']:
        """
        Compute the best card to play based on dynamic synergies and the current game state.
        """
        logger.info(f"Computing synergies for cards: {[card.name for card in cards]}")
        if not cards:
            return None

        # Generate dynamic formulas based on the current game state
        dynamic_formulas = self.generate_dynamic_formulas(game_state)

        # Select the best card based on dynamic synergy score calculation
        best_card = max(cards, key=lambda card: self._compute_synergy_score(card, dynamic_formulas))
        logger.info(f"Best card based on dynamic synergy: {best_card.name}")
        return best_card

    def _compute_synergy_score(self, card: 'Card', formulas: List[sp.Eq]) -> float:
        """
        Compute a synergy score for a card using dynamically generated formulas.
        """
        score = 0.0
        try:
            mana_value = card.calculate_mana_value()

            # Evaluate formulas and adjust score accordingly
            for formula in formulas:
                # Example: If a formula maximizes health difference, give priority to damage-dealing cards
                if formula.has(sp.Symbol('health')) or formula.has(sp.Symbol('opponent_health')):
                    score += self._evaluate_health_difference_synergy(card, formula)
                elif formula.has(sp.Symbol('mana')):
                    score += self._evaluate_mana_efficiency(card, formula, mana_value)
                elif formula.has(sp.Symbol('creatures')):
                    score += self._evaluate_board_presence(card, formula)

        except (ValueError, ZeroDivisionError):
            pass

        return score

    def _creature_synergy_score(self, card: 'Card') -> float:
        """
        Calculate synergy score for creatures based on power, toughness, and abilities.
        """
        score = 0
        power = int(card.power) if card.power and card.power.isdigit() else 0
        toughness = int(card.toughness) if card.toughness and card.toughness.isdigit() else 0

        # Weigh creatures with higher power/toughness more
        score += power + toughness

        # Synergies based on abilities (e.g., flying, deathtouch)
        if "Flying" in card.oracle_text:
            score += 2
        if "Deathtouch" in card.oracle_text:
            score += 3
        if "Lifelink" in card.oracle_text:
            score += 2

        return score

    def _enchantment_synergy_score(self, card: 'Card') -> float:
        """
        Calculate synergy score for enchantments.
        """
        # Example: Enchantments that buff creatures or provide lasting effects are valuable.
        score = 5
        if "All creatures" in card.oracle_text:
            score += 3  # Global effects are strong
        return score

    def _spell_synergy_score(self, card: 'Card') -> float:
        """
        Calculate synergy score for instants and sorceries.
        """
        # Instants and sorceries that directly affect the board (damage, destroy, draw) are valuable
        score = 0
        if "Draw" in card.oracle_text:
            score += 3  # Drawing cards is always strong
        if "Destroy" in card.oracle_text:
            score += 4  # Destroying creatures or permanents is high value
        if "Damage" in card.oracle_text:
            score += 2  # Dealing direct damage is useful

        return score

    def _land_synergy_score(self, card: 'Card') -> float:
        """
        Calculate synergy score for lands.
        """
        # Lands that generate more than one mana or have abilities are more valuable.
        score = 1  # Basic lands are always useful
        if "Add two mana" in card.oracle_text:
            score += 3  # Lands that add extra mana are very valuable
        if "Untap" in card.oracle_text:
            score += 2  # Lands that untap or have other abilities

        return score

    def _planeswalker_synergy_score(self, card: 'Card') -> float:
        """
        Calculate synergy score for planeswalkers based on their abilities.
        """
        score = 8  # Base score for Planeswalkers

        # Check oracle text for abilities related to Planeswalkers
        if "Draw" in card.oracle_text:
            score += 4  # Card draw ability
        if "Destroy" in card.oracle_text:
            score += 4  # Destruction ability
        if "Add loyalty" in card.oracle_text:
            score += 3  # Loyalty increasing ability
        if "Exile" in card.oracle_text:
            score += 4  # Exile effects are strong

        return score

# --- NEURAL NETWORK ENGINE: For Card Recognition & Analysis ---
class AttentionAggregation(nn.Module):
    def __init__(self, latent_dim, dropout=0.1):
        super(AttentionAggregation, self).__init__()
        self.self_attention = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True, dropout=dropout).to(device)
        self.cross_attention = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True, dropout=dropout).to(device)
        self.layer_norm1 = nn.LayerNorm(latent_dim).to(device)
        self.layer_norm2 = nn.LayerNorm(latent_dim).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, self_features, other_features):
        # Self-attention with residual connection and dropout
        self_attn_output, _ = self.self_attention(self_features, self_features, self_features)
        self_attn_output = self.dropout(self_attn_output)
        self_attn_output = self.layer_norm1(self_attn_output + self_features)

        # Cross-attention with residual connection and dropout
        cross_attn_output, _ = self.cross_attention(self_attn_output, other_features, other_features)
        cross_attn_output = self.dropout(cross_attn_output)
        cross_attn_output = self.layer_norm2(cross_attn_output + self_attn_output)

        return cross_attn_output

# --- REINFORCEMENT LEARNING ENGINE: For AI Learning & Adaptation ---
class MTGAI(nn.Module):
    """
    AI Model that learns optimal plays through reinforcement learning.
    """
    def __init__(self, input_size, hidden_size, output_size, latent_dim):
        super(MTGAI, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).to(device)
        self.fc2 = nn.Linear(hidden_size, latent_dim).to(device)
        self.attention_aggregation = AttentionAggregation(latent_dim).to(device)
        self.output_layer = nn.Linear(latent_dim, output_size).to(device)

    def forward(self, self_features, other_features):
        # Process self features through fully connected layers
        x = torch.relu(self.fc1(self_features))
        x = torch.relu(self.fc2(x))

        # Process other features through the same layers
        other_x = torch.relu(self.fc1(other_features))
        other_x = torch.relu(self.fc2(other_x))

        # Adjust dimensions for attention layers
        x = x.unsqueeze(1)  # Shape: (batch_size, seq_len=1, latent_dim)
        other_x = other_x.unsqueeze(1)  # Shape: (batch_size, seq_len=1, latent_dim)

        # Incorporate attention aggregation (self vs. opponent features)
        attn_output = self.attention_aggregation(x, other_x)

        # Flatten and pass through the output layer
        attn_output = attn_output.squeeze(1)  # Shape: (batch_size, latent_dim)
        output = self.output_layer(attn_output)

        return output


# --- AGENT ENGINE: For AI Decision-Making & Strategy ---
class Agent:
    def __init__(
        self,
        state_size,
        action_size,
        latent_dim,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        update_target_every=100,
        batch_size=32
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_target_every = update_target_every
        self.batch_size = batch_size
        self.step_count = 0

        self.q_network = MTGAI(
            input_size=state_size,
            hidden_size=128,
            output_size=action_size,
            latent_dim=latent_dim
        ).to(device)

        self.target_q_network = MTGAI(
            input_size=state_size,
            hidden_size=128,
            output_size=action_size,
            latent_dim=latent_dim
        ).to(device)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = []  # To store experiences for training
        self.criterion = nn.MSELoss()

    def take_action_and_calculate_reward(self, player: 'Player', opponent: 'Player', game_stack: 'Stack'):
        # Extract features
        state_self = player._extract_features()
        state_opponent = opponent._extract_features()

        # Determine valid actions
        valid_actions = []
        for i, card in enumerate(player.hand):
            if player.can_play_card(card):
                valid_actions.append(i)
        # Always add the 'do nothing' action
        valid_actions.append(self.action_size - 1)

        # Choose action
        if random.random() < self.epsilon:
            action = random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_self.unsqueeze(0), state_opponent.unsqueeze(0))
                q_values = q_values.squeeze(0)  # Shape: (action_size,)
                # Mask invalid actions
                mask = torch.full((self.action_size,), float('-inf')).to(device)
                mask[valid_actions] = 0
                q_values = q_values + mask
                action = torch.argmax(q_values).item()

        # Execute action
        if action == self.action_size - 1:
            # Do nothing
            player.took_action = False
        else:
            success = player.play_card_by_index(action, game_stack)
            if not success:
                player.took_action = False

        # Calculate reward
        reward = self.calculate_reward(player, opponent)
        # Store experience
        next_state_self = player._extract_features()
        next_state_opponent = opponent._extract_features()
        done = player.has_lost() or opponent.has_lost()
        self.replay_buffer.append((state_self, state_opponent, action, reward, next_state_self, next_state_opponent, done))
        # Train the network
        self.train_q_network()
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def calculate_reward(self, player, opponent):
        """
        Reward function that calculates a numerical reward based on the player's game state.
        """
        reward = 0

        # Reward for playing a card
        if player.battlefield != player.previous_battlefield:
            reward += 10  # Reward for improving board state

        # Reward for playing a land
        if player.lands_played_this_turn > player.previous_lands_played:
            reward += 5  # Reward for playing lands

        # Reward for dealing damage
        damage_dealt = opponent.previous_health - opponent.health
        if damage_dealt > 0:
            reward += damage_dealt * 5  # Reward for dealing damage

        # Penalize for unspent mana when playable cards are available
        if sum(player.mana_pool.values()) > 0 and player.has_playable_cards():
            reward -= 5

        # Penalize for inaction when actions are available
        if not player.took_action and player.has_playable_cards():
            reward -= 1

        # Reward for winning the game
        if opponent.has_lost():
            reward += 100

        # Penalize for losing the game
        if player.has_lost():
            reward -= 100

        return reward

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train_q_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train
        # Sample a batch of experiences
        batch = random.sample(self.replay_buffer, self.batch_size)
        # Unpack the batch
        states_self, states_opponent, actions, rewards, next_states_self, next_states_opponent, dones = zip(*batch)
        # Convert to tensors
        states_self = torch.stack(states_self).to(device)
        states_opponent = torch.stack(states_opponent).to(device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states_self = torch.stack(next_states_self).to(device)
        next_states_opponent = torch.stack(next_states_opponent).to(device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        # Compute current Q values
        current_q_values = self.q_network(states_self, states_opponent).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states_self, next_states_opponent).max(1)[0]
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.update_target_network()



from enum import Enum
from typing import List, Optional

class CardType(Enum):
    CREATURE = "Creature"
    LAND = "Land"
    ENCHANTMENT = "Enchantment"
    INSTANT = "Instant"
    SORCERY = "Sorcery"
    PLANESWALKER = "Planeswalker"

# --- CARD CLASS ---
class Card:
    def __init__(
        self,
        name: str,
        card_type: str,
        mana_cost: str,
        power: Optional[str] = None,
        toughness: Optional[str] = None,
        colors: List[str] = [],
        keywords: List[str] = [],
        oracle_text: str = '',
        image_data: Optional[np.ndarray] = None,
        summoning_sick: bool = True
    ):
        self.name = name
        self.card_type = card_type
        self.mana_cost = mana_cost
        self.power = power
        self.toughness = toughness
        self.colors = colors
        self.keywords = keywords
        self.oracle_text = oracle_text
        self.image_data = image_data
        self.summoning_sick = summoning_sick
        self.image = None
        self.tapped = False
        self.owner: Optional['Player'] = None

    def __str__(self) -> str:
        pt = f"{self.power}/{self.toughness}" if self.power and self.toughness else ""
        return f"{self.name} ({self.card_type}) - Cost: {self.mana_cost} {pt}"

    @staticmethod
    def from_parquet(card_data: pd.Series) -> 'Card':
        try:
            name = card_data.get('name', 'Unknown')
            card_type = card_data.get('type_line', 'Unknown')
            mana_cost = str(card_data.get('mana_cost', ''))  # Ensure mana_cost is always a string
            power = card_data.get('power')
            toughness = card_data.get('toughness')
            colors = card_data.get('colors', [])
            if colors is None:
                colors = []  # If colors is None, replace it with an empty list
            keywords = card_data.get('keywords', [])
            oracle_text = card_data.get('oracle_text', '')

            # Ensure image data is treated as bytes
            image_data = card_data.get('image_data')
            if isinstance(image_data, bytes):
                pass  # Image data is already in bytes
            elif isinstance(image_data, str):
                # If image data is a base64-encoded string, decode it
                image_data = base64.b64decode(image_data)
            elif isinstance(image_data, np.ndarray):
                # Convert numpy array to bytes
                image_data = image_data.tobytes()
            else:
                # Handle other formats if necessary
                image_data = None

            return Card(name, card_type, mana_cost, power, toughness, colors, keywords, oracle_text, image_data)
        except Exception as e:
            logger.error(f"Error creating Card from Parquet: {e}")
            raise

    def load_image(self):
        """
        Loads the card image from flattened data for display, resizing it to a fixed size.
        """
        if self.image_data is not None:
            try:
                # Decode the image data to an image
                image_array = np.frombuffer(self.image_data, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

                if image is not None:
                    # Resize image to the desired size (e.g., 60x90)
                    resized_image = cv2.resize(image, (60, 90))

                    # Convert image from BGR to RGB for pygame compatibility
                    if resized_image.shape[2] == 4:
                        # Handle images with alpha channel
                        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2RGBA)
                    else:
                        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

                    # Convert to pygame Surface
                    mode = 'RGBA' if resized_image.shape[2] == 4 else 'RGB'
                    self.image = pygame.image.frombuffer(image_rgb.tobytes(), image_rgb.shape[1::-1], mode)
                else:
                    raise ValueError(f"Error decoding image data for card {self.name}.")
            except Exception as e:
                logger.error(f"Error loading image for card {self.name}: {e}")
                self.image = None  # Use a placeholder or empty surface
        else:
            logger.warning(f"No image data for card {self.name}.")
            self.image = None  # Use a placeholder or empty surface

    def calculate_mana_value(self) -> int:
        """
        Calculates the converted mana cost (CMC).
        """
        mana_symbols = re.findall(r'\{(.*?)\}', self.mana_cost)
        mana_value = 0
        for symbol in mana_symbols:
            if symbol.isdigit():
                mana_value += int(symbol)
            elif symbol == 'X':
                # X is variable; handle this case in gameplay logic (assumed as 0 for simplicity)
                mana_value += 0
            elif '/' in symbol:
                # Hybrid mana counts as 1
                mana_value += 1
            else:
                # Colored mana symbols count as 1
                mana_value += 1
        return mana_value

    def get_mana_cost(self) -> str:
        """
        Returns the mana cost of the card.
        """
        return self.mana_cost

    def get_mana_cost_as_list(self) -> List[str]:
        """
        Returns the mana cost of the card as a list of individual mana symbols.
        """
        mana_symbols = re.findall(r'\{(.*?)\}', self.mana_cost)
        return mana_symbols
    
    def get_mana_requirements(self) -> List[List[str]]:
        """
        Parses the mana cost and returns a list of possible mana combinations.
        Each element in the list represents a mana symbol's possible colors.
        """
        mana_symbols = re.findall(r'\{(.*?)\}', self.mana_cost)
        mana_requirements: List[List[str]] = []
        for symbol in mana_symbols:
            if symbol.isdigit():
                # Generic mana cost can be paid with any mana
                mana_requirements.extend([['Any']] * int(symbol))
            elif symbol == 'C':
                # Colorless mana requirement
                mana_requirements.append(['Colorless'])
            elif symbol == 'X':
                # 'X' can be any amount of mana; handle this in game logic
                pass  # For now, ignore or set to zero
            elif symbol == 'T':
                # 'T' is the tap symbol and not part of mana costs
                continue
            elif '/' in symbol:
                # Hybrid mana symbols like {G/W}
                options = symbol.split('/')
                mana_requirements.append(options)
            else:
                # Regular colored mana symbol
                mana_requirements.append([symbol])
        return mana_requirements

    def tap(self) -> None:
        """
        Taps the card.
        """
        if not self.tapped:
            self.tapped = True
            logger.info(f"{self.name} has been tapped.")
        else:
            logger.warning(f"{self.name} is already tapped.")

    def untap(self) -> None:
        """
        Untaps the card.
        """
        if self.tapped:
            self.tapped = False
            self.summoning_sick = False
            logger.info(f"{self.name} has been untapped.")

    def is_playable(self, mana_pool: Dict[str, int]) -> bool:
        """
        Determines if the card can be played with the available mana.
        """
        mana_requirements = self.get_mana_requirements()
        return self._can_pay_mana(mana_requirements, mana_pool.copy())

    def _can_pay_mana(self, mana_requirements: List[List[str]], mana_pool: Dict[str, int]) -> bool:
        """
        Helper method to recursively determine if mana costs can be paid.
        """
        if not mana_requirements:
            return True  # All mana costs have been paid
        for color_option in mana_requirements[0]:
            temp_mana_pool = mana_pool.copy()
            if color_option == 'Any' or color_option == 'Colorless':
                # Try paying with any available mana
                for color in temp_mana_pool:
                    if temp_mana_pool[color] > 0:
                        temp_mana_pool[color] -= 1
                        if self._can_pay_mana(mana_requirements[1:], temp_mana_pool):
                            return True
            else:
                if temp_mana_pool.get(color_option, 0) > 0:
                    temp_mana_pool[color_option] -= 1
                    if self._can_pay_mana(mana_requirements[1:], temp_mana_pool):
                        return True
        return False  # Cannot pay this mana symbol

    def is_creature(self) -> bool:
        """
        Checks if the card is a creature.
        """
        return 'Creature' in self.card_type

    def is_land(self) -> bool:
        """
        Checks if the card is a land.
        """
        return 'Land' in self.card_type

    def has_keyword(self, keyword: str) -> bool:
        """
        Checks if the card has a specific keyword.
        """
        return keyword in self.keywords

    def get_possible_mana(self) -> List[str]:
        """
        Returns a list of possible mana types that this card can produce when tapped.
        """
        mana_types = []
        if self.is_land():
            # Handle basic lands
            if 'Plains' in self.name:
                mana_types.append('W')
            elif 'Island' in self.name:
                mana_types.append('U')
            elif 'Swamp' in self.name:
                mana_types.append('B')
            elif 'Mountain' in self.name:
                mana_types.append('R')
            elif 'Forest' in self.name:
                mana_types.append('G')
            else:
                # For non-basic lands, parse the oracle text to find mana abilities
                oracle_text = self.oracle_text if self.oracle_text else ''
                # Look for abilities like "{T}: Add {G} or {R}."
                mana_abilities = re.findall(r'\{T\}: Add (.*?)\.', oracle_text)
                for ability in mana_abilities:
                    # Extract mana symbols
                    mana_symbols = re.findall(r'\{(.*?)\}', ability)
                    mana_types.extend(mana_symbols)
        return mana_types

    def produce_mana(self, chosen_mana: str) -> Dict[str, int]:
        """
        Produces the specified mana when the card is tapped.
        """
        mana_produced = {}
        possible_mana = self.get_possible_mana()
        if chosen_mana in possible_mana:
            mana_produced[chosen_mana] = 1
        else:
            logger.error(f"{self.name} cannot produce mana of type {chosen_mana}.")
        return mana_produced

    def requires_life_to_activate(self) -> bool:
        """
        Checks if tapping this card for mana requires the player to pay life.
        """
        return 'Pay 1 life' in self.oracle_text or 'lose 1 life' in self.oracle_text

    def life_cost_to_activate(self) -> int:
        """
        Returns the life cost required to activate this card's mana ability.
        """
        if 'Pay 1 life' in self.oracle_text or 'lose 1 life' in self.oracle_text:
            return 1
        return 0

class CardEffectHandler:
    @staticmethod
    def apply_effect(card: Card, game_state: 'GameState'):
        """
        Applies the effect of the card based on its type.
        """
        for card_type in card.card_types:
            if card_type == CardType.CREATURE:
                CardEffectHandler._apply_creature_effect(card, game_state)
            elif card_type == CardType.LAND:
                CardEffectHandler._apply_land_effect(card, game_state)
            elif card_type == CardType.ENCHANTMENT:
                CardEffectHandler._apply_enchantment_effect(card, game_state)
            elif card_type == CardType.INSTANT:
                CardEffectHandler._apply_instant_effect(card, game_state)
            elif card_type == CardType.SORCERY:
                CardEffectHandler._apply_sorcery_effect(card, game_state)
            elif card_type == CardType.PLANESWALKER:
                CardEffectHandler._apply_planeswalker_effect(card, game_state)

    @staticmethod
    def _apply_creature_effect(card: Card, game_state: 'GameState'):
        # Logic for creatures entering the battlefield
        game_state.add_to_battlefield(card)

    @staticmethod
    def _apply_land_effect(card: Card, game_state: 'GameState'):
        # Logic for playing lands
        game_state.add_land(card)

    @staticmethod
    def _apply_enchantment_effect(card: Card, game_state: 'GameState'):
        # Logic for enchantments
        game_state.add_enchantment(card)

    @staticmethod
    def _apply_instant_effect(card: Card, game_state: 'GameState'):
        # Logic for instant spells
        game_state.resolve_instant(card)

    @staticmethod
    def _apply_sorcery_effect(card: Card, game_state: 'GameState'):
        # Logic for sorcery spells
        game_state.resolve_sorcery(card)

    @staticmethod
    def _apply_planeswalker_effect(card: Card, game_state: 'GameState'):
        # Logic for planeswalkers
        game_state.add_planeswalker(card)


# --- DECK CLASS ---
class Deck:
    """
    Represents a deck of cards.
    """
    def __init__(self, cards: List[Card]):
        if len(cards) < 60:
            raise ValueError("Deck must contain at least 60 cards.")
        self.cards = cards
        self.shuffle()

    def shuffle(self) -> None:
        """
        Shuffles the deck.
        """
        random.shuffle(self.cards)
        logger.info("Deck has been shuffled.")

    def draw(self) -> Optional[Card]:
        """
        Draws a card from the top of the deck.
        """
        if self.cards:
            card = self.cards.pop()
            logger.info(f"Drew card: {card.name}")
            return card
        else:
            logger.warning("Attempted to draw from an empty deck.")
            return None

    def is_empty(self) -> bool:
        """
        Checks if the deck is empty.
        """
        return len(self.cards) == 0

# --- PLAYER CLASS ---
class Player:
    def __init__(self, name: str, deck: 'Deck'):
        self.name = name
        self.deck = deck
        self.hand: List['Card'] = []
        self.battlefield: List['Card'] = []
        self.graveyard: List['Card'] = []
        self.exile_pile: List['Card'] = []
        self.mana_pool: Dict[str, int] = {}
        self.lands_played_this_turn = 0
        self.health = 20
        self.max_hand_size = 7
        self.took_action = False
        # For tracking previous state
        self.previous_health = self.health
        self.previous_creature_count = 0
        self.previous_board_state_value = 0
        self.previous_battlefield = []
        self.previous_lands_played = 0

    def _extract_features(self) -> torch.Tensor:
        """
        Extract features from the player's current state for use by the AI agent.
        """
        # Player's health (normalized)
        health_feature = torch.tensor([self.health / 20.0], dtype=torch.float32)

        # Number of cards in hand (normalized)
        hand_feature = torch.tensor([len(self.hand) / 7.0], dtype=torch.float32)

        # Number of creatures on the battlefield (normalized)
        creatures_on_battlefield = len([card for card in self.battlefield if card.is_creature()])
        battlefield_feature = torch.tensor([creatures_on_battlefield / 10.0], dtype=torch.float32)

        # Mana pool (representation for each color)
        mana_colors = ['W', 'U', 'B', 'R', 'G', 'Colorless']
        mana_feature = torch.tensor([self.mana_pool.get(color, 0) / 10.0 for color in mana_colors], dtype=torch.float32)

        # Concatenate all features into a single tensor
        features = torch.cat([health_feature, hand_feature, battlefield_feature, mana_feature])

        return features

    def evaluate_board_state(self) -> int:
        """
        Evaluates the current board state and returns a numerical score representing the player's advantage.
        """
        board_value = 0

        # Value each creature on the battlefield
        for card in self.battlefield:
            if card.is_creature():
                power = int(card.power) if card.power and card.power.isdigit() else 0
                toughness = int(card.toughness) if card.toughness and card.toughness.isdigit() else 0
                board_value += (power + toughness) * 2  # Adjust weights as necessary

        # Value lands
        num_lands = len([card for card in self.battlefield if card.is_land()])
        board_value += num_lands * 1  # Adjust weight based on importance

        # Value other permanents
        num_enchantments = len([card for card in self.battlefield if 'Enchantment' in card.card_type])
        num_artifacts = len([card for card in self.battlefield if 'Artifact' in card.card_type])
        num_planeswalkers = len([card for card in self.battlefield if 'Planeswalker' in card.card_type])

        board_value += num_enchantments * 3
        board_value += num_artifacts * 2
        board_value += num_planeswalkers * 5

        # Include life total
        board_value += self.health * 1  # Adjust weight if necessary

        return board_value

    def draw_card(self) -> None:
        """
        Player draws a card from their deck.
        """
        card = self.deck.draw()
        if card:
            self.hand.append(card)
            logger.info(f"{self.name} drew a card: {card.name}")
            logger.info(f"{self.name}'s current hand: {[c.name for c in self.hand]}")
            card.load_image()
        else:
            logger.error(f"{self.name} cannot draw a card. Deck is empty.")
            self.lose_game()

    def play_card(self, card: 'Card') -> None:
        """
        Plays a card from the player's hand.
        """
        if card in self.hand:
            if self.can_play_card(card):
                if card.is_land():
                    if self.lands_played_this_turn < 1:
                        self.hand.remove(card)
                        self.battlefield.append(card)
                        self.lands_played_this_turn += 1
                        logger.info(f"{self.name} plays land: {card.name}")
                    else:
                        logger.info(f"{self.name} cannot play more lands this turn.")
                else:
                    # Add mana until cost is met
                    self.add_mana_until_cost_met(card)
                    if self.is_playable(card):
                        self.spend_mana(card)
                        self.hand.remove(card)
                        self.game_stack.add_to_stack(StackItem(self, card))
                        logger.info(f"{self.name} casts {card.name}.")
                    else:
                        logger.warning(f"{self.name} cannot cast {card.name} due to insufficient mana.")
            else:
                logger.warning(f"{self.name} cannot play {card.name} due to mana requirements or other restrictions.")
        else:
            logger.error(f"{self.name} does not have {card.name} in their hand.")

    def is_playable(self, card: 'Card') -> bool:
        mana_requirements = card.get_mana_requirements()
        temp_mana_pool = self.mana_pool.copy()
        return self._can_pay_mana(mana_requirements, temp_mana_pool)

    def has_playable_cards(self) -> bool:
        """
        Checks if the player has any cards in hand that can be played.
        """
        for card in self.hand:
            if self.can_play_card(card):
                return True
        return False

    def can_play_card(self, card: 'Card') -> bool:
        """
        Determines if the card can be played, considering lands played and mana availability.
        """
        if card.is_land():
            return self.lands_played_this_turn < 1
        else:
            return self.can_produce_mana_to_pay_for(card)

    def can_produce_mana_to_pay_for(self, card: 'Card') -> bool:
        """
        Determines if the player can produce enough mana by tapping available lands to pay for the card.
        """
        mana_requirements = card.get_mana_requirements()
        mana_pool = self.mana_pool.copy()
        untapped_lands = [land for land in self.battlefield if land.is_land() and not land.tapped]

        # Simulate tapping lands to produce mana
        potential_mana_pool = mana_pool.copy()
        for land in untapped_lands:
            possible_mana = land.get_possible_mana()
            for mana in possible_mana:
                potential_mana_pool[mana] = potential_mana_pool.get(mana, 0) + 1
                break  # Assuming land produces one mana per tap

        # Now, attempt to pay mana requirements with potential_mana_pool
        return self._can_pay_mana(mana_requirements, potential_mana_pool)

    def spend_mana(self, card: 'Card') -> None:
        """
        Deducts mana from the player's mana pool to pay for a card.
        """
        mana_requirements = card.get_mana_requirements()
        if not self._pay_mana(mana_requirements, self.mana_pool.copy()):
            logger.error(f"{self.name} could not pay mana cost for {card.name}.")
        else:
            logger.info(f"{self.name} spent mana to cast {card.name}. Remaining mana pool: {self.mana_pool}")

    def _pay_mana(self, mana_requirements: List[List[str]], mana_pool: Dict[str, int]) -> bool:
        """
        Attempts to pay mana costs using the player's mana pool.
        """
        if not mana_requirements:
            self.mana_pool = mana_pool  # Update the actual mana pool after successful payment
            return True
        for color_option in mana_requirements[0]:
            temp_mana_pool = mana_pool.copy()
            if color_option == 'Any' or color_option == 'Colorless':
                # Try paying with any available mana
                for color in temp_mana_pool:
                    if temp_mana_pool[color] > 0:
                        temp_mana_pool[color] -= 1
                        if self._pay_mana(mana_requirements[1:], temp_mana_pool):
                            return True
            else:
                if temp_mana_pool.get(color_option, 0) > 0:
                    temp_mana_pool[color_option] -= 1
                    if self._pay_mana(mana_requirements[1:], temp_mana_pool):
                        return True
        return False  # Cannot pay this mana symbol

    def choose_mana_to_produce(self, possible_mana: List[str]) -> str:
        """
        Chooses which mana to produce from possible options.
        For base Player, default to first option.
        """
        logger.info(f"{self.name} is tapping a land to produce mana.")
        chosen_mana = possible_mana[0]  # Default to the first available mana type
        logger.info(f"{self.name} chooses to produce {chosen_mana} mana.")
        return chosen_mana

    def _can_pay_mana(self, mana_requirements: List[List[str]], temp_mana_pool: Dict[str, int]) -> bool:
        if not mana_requirements:
            return True
        for color_option in mana_requirements[0]:
            temp_mana_pool_copy = temp_mana_pool.copy()
            if color_option == 'Any' or color_option == 'Colorless':
                # Try paying with any available mana
                for color in temp_mana_pool_copy:
                    if temp_mana_pool_copy[color] > 0:
                        temp_mana_pool_copy[color] -= 1
                        if self._can_pay_mana(mana_requirements[1:], temp_mana_pool_copy):
                            return True
            else:
                if temp_mana_pool_copy.get(color_option, 0) > 0:
                    temp_mana_pool_copy[color_option] -= 1
                    if self._can_pay_mana(mana_requirements[1:], temp_mana_pool_copy):
                        return True
        return False

    def add_mana_until_cost_met(self, card: 'Card'):
        """
        Taps lands to add mana until the cost of the given card is met or all lands are tapped.
        """
        untapped_lands = [land for land in self.battlefield if land.is_land() and not land.tapped]

        while not card.is_playable(self.mana_pool) and untapped_lands:
            land = untapped_lands.pop(0)
            possible_mana = land.get_possible_mana()
            chosen_mana = self.choose_mana_to_produce(possible_mana)
            mana_produced = land.produce_mana(chosen_mana)
            if mana_produced:
                # Check for life cost if applicable
                life_cost = land.life_cost_to_activate()
                if life_cost > 0:
                    self.health -= life_cost
                    logger.info(f"{self.name} pays {life_cost} life to activate {land.name}.")
                    logger.info(f"{self.name} is now at {self.health} health.")

                land.tap()  # The land is now tapped

                # Add the produced mana to the player's mana pool
                for color, amount in mana_produced.items():
                    self.mana_pool[color] = self.mana_pool.get(color, 0) + amount
                logger.info(f"{self.name} adds {amount} {color} mana to their mana pool by tapping {land.name}.")
                logger.info(f"Mana pool after tapping {land.name}: {self.mana_pool}")
            else:
                logger.warning(f"{self.name} could not produce mana with {land.name}.")

    def untap_all(self) -> None:
        """
        Untaps all tapped cards on the battlefield and resets summoning sickness.
        """
        for card in self.battlefield:
            card.untap()
            if card.is_creature():
                card.summoning_sick = False  # Creatures lose summoning sickness after a full turn
        logger.info(f"{self.name} untapped all cards on the battlefield.")
        self.lands_played_this_turn = 0

    def lose_game(self) -> None:
        """
        Sets the player's health to zero, indicating they've lost the game.
        """
        self.health = 0
        logger.info(f"{self.name} has lost the game.")

    def has_lost(self) -> bool:
        """
        Checks if the player has lost the game.
        """
        return self.health <= 0

    def declare_attackers(self, players: List['Player']) -> List['Card']:
        """
        Declares attackers during the combat phase.
        """
        attackers = [card for card in self.battlefield if card.is_creature() and not card.tapped and not card.summoning_sick]
        for attacker in attackers:
            attacker.tap()
            logger.info(f"{self.name} declares {attacker.name} as an attacker.")
        if not attackers:
            logger.info(f"{self.name} has no creatures to attack with.")
        return attackers

    def declare_blockers(self, attackers: List['Card']) -> Dict['Card', Optional['Card']]:
        """
        Declares blockers during the combat phase.
        """
        blockers: Dict['Card', Optional['Card']] = {}
        available_blockers = [card for card in self.battlefield if card.is_creature() and not card.tapped]

        # Simple AI: Block attackers with available blockers
        for attacker in attackers:
            if available_blockers:
                blocker = available_blockers.pop(0)
                blocker.tap()
                logger.info(f"{self.name} blocks {attacker.name} with {blocker.name}.")
                blockers[attacker] = blocker
            else:
                # No blockers available, attacker is unblocked
                blockers[attacker] = None
        return blockers

    def resolve_combat(self, blockers: Dict['Card', Optional['Card']], opponent: 'Player') -> None:
        """
        Resolves combat damage between attackers and blockers.
        """
        for attacker, blocker in blockers.items():
            attacker_power = self.get_creature_power(attacker)
            if blocker:
                blocker_toughness = opponent.get_creature_toughness(blocker)
                blocker_power = opponent.get_creature_power(blocker)
                attacker_toughness = self.get_creature_toughness(attacker)
                logger.info(f"{attacker.name} ({attacker_power}/{attacker_toughness}) is blocked by {blocker.name} ({blocker_power}/{blocker_toughness})")
                if attacker_power >= blocker_toughness:
                    opponent.move_to_graveyard(blocker)
                    opponent.battlefield.remove(blocker)
                    logger.info(f"{blocker.name} is destroyed.")
                if blocker_power >= attacker_toughness:
                    self.move_to_graveyard(attacker)
                    self.battlefield.remove(attacker)
                    logger.info(f"{attacker.name} is destroyed.")
            else:
                opponent.receive_damage(attacker_power)
                logger.info(f"{attacker.name} deals {attacker_power} damage to {opponent.name}.")

    def get_opponent(self, players: List['Player']) -> 'Player':
        """
        Finds the opponent from the list of players.
        """
        for player in players:
            if player != self:
                return player
        raise ValueError("Opponent not found")

    def get_creature_power(self, creature: 'Card') -> int:
        """
        Returns the creature's power as an integer.
        """
        try:
            if creature.power == 'X':
                return 0
            return int(creature.power)
        except (ValueError, TypeError):
            return 0

    def get_creature_toughness(self, creature: 'Card') -> int:
        """
        Returns the creature's toughness as an integer.
        """
        try:
            if creature.toughness == 'X':
                return 0
            return int(creature.toughness)
        except (ValueError, TypeError):
            return 0

    def move_to_graveyard(self, card: 'Card') -> None:
        """
        Moves a card to the graveyard.
        """
        self.graveyard.append(card)
        logger.info(f"{card.name} has been moved to the graveyard.")

    def receive_damage(self, amount: int) -> None:
        """
        Reduces the player's health by the specified damage amount.
        """
        self.health -= amount
        logger.info(f"{self.name} received {amount} damage and is now at {self.health} health.")
        if self.health <= 0:
            logger.info(f"{self.name} has lost the game.")

    def discard_down_to_max_hand_size(self) -> None:
        """
        Discards cards down to the maximum hand size at the end of the turn.
        """
        while len(self.hand) > self.max_hand_size:
            discarded_card = self.hand.pop()
            self.move_to_graveyard(discarded_card)
            logger.info(f"{self.name} discards {discarded_card.name} to meet the maximum hand size.")

    def empty_mana_pool(self) -> None:
        """
        Empties the player's mana pool at the end of each phase.
        """
        self.mana_pool.clear()
        logger.info(f"{self.name}'s mana pool has been emptied.")

    def __str__(self) -> str:
        return f"Player {self.name}: {self.health} HP, Mana Pool: {self.mana_pool}"


class AIPlayer(Player):
    """
    Represents an AI-controlled player with enhanced decision-making using attention mechanisms.
    """
    def __init__(self, name: str, deck: 'Deck', agent: 'Agent'):
        super().__init__(name, deck)
        self.agent = agent

    def _extract_features(self) -> torch.Tensor:
        """
        Extract features from the player's current state for use by the AI agent.
        """
        health_feature = torch.tensor([self.health / 20.0], dtype=torch.float32, device=device)
        hand_feature = torch.tensor([len(self.hand) / 7.0], dtype=torch.float32, device=device)
        creatures_on_battlefield = len([card for card in self.battlefield if card.is_creature()])
        battlefield_feature = torch.tensor([creatures_on_battlefield / 10.0], dtype=torch.float32, device=device)
        mana_colors = ['W', 'U', 'B', 'R', 'G', 'Colorless']
        mana_feature = torch.tensor(
            [self.mana_pool.get(color, 0) / 10.0 for color in mana_colors],
            dtype=torch.float32,
            device=device
        )
        features = torch.cat([health_feature, hand_feature, battlefield_feature, mana_feature])
        return features

    def play_lands(self) -> bool:
        """
        AI decides to play a land if possible.
        """
        if self.lands_played_this_turn < 1:
            for i, card in enumerate(self.hand):
                if card.is_land():
                    self.hand.pop(i)
                    self.battlefield.append(card)
                    self.lands_played_this_turn += 1
                    logger.info(f"{self.name} plays land: {card.name}")
                    self.took_action = True
                    return True
        return False

    def play_card_by_index(self, card_index: int, game_stack: 'Stack') -> bool:
        if 0 <= card_index < len(self.hand):
            card = self.hand[card_index]
            if self.can_play_card(card):
                if card.is_land():
                    if self.lands_played_this_turn < 1:
                        self.hand.pop(card_index)
                        self.battlefield.append(card)
                        self.lands_played_this_turn += 1
                        logger.info(f"{self.name} plays land: {card.name}")
                        self.took_action = True
                        return True
                    else:
                        logger.info(f"{self.name} cannot play more lands this turn.")
                        return False
                else:
                    self.add_mana_until_cost_met(card)
                    if self.is_playable(card):
                        self.spend_mana(card)
                        self.hand.pop(card_index)
                        game_stack.add_to_stack(StackItem(self, card))
                        logger.info(f"{self.name} casts {card.name}.")
                        self.took_action = True
                        return True
                    else:
                        logger.info(f"{self.name} cannot cast {card.name} due to insufficient mana.")
                        return False
            else:
                logger.info(f"{self.name} cannot play {card.name} due to insufficient mana or other restrictions.")
                return False
        else:
            logger.info(f"{self.name} has no card at index {card_index}.")
            self.took_action = False
            return False

    def choose_mana_to_produce(self, possible_mana: List[str]) -> str:
        """
        AI chooses which mana to produce from possible options.
        """
        needed_mana = self.get_needed_mana_types()
        for mana in needed_mana:
            if mana in possible_mana:
                return mana
        return random.choice(possible_mana)

    def get_needed_mana_types(self) -> List[str]:
        """
        Returns a list of mana types that are required to play cards in hand.
        """
        needed_mana = set()
        for card in self.hand:
            if not card.is_playable(self.mana_pool):
                mana_requirements = card.get_mana_requirements()
                for mana_options in mana_requirements:
                    for mana in mana_options:
                        if mana in ['Any', 'Colorless']:
                            needed_mana.update(['W', 'U', 'B', 'R', 'G'])
                        else:
                            needed_mana.add(mana)
        return list(needed_mana) if needed_mana else ['W', 'U', 'B', 'R', 'G']

    def add_mana_until_cost_met(self, card: 'Card'):
        """
        Taps lands to add mana until the cost of the given card is met or all lands are tapped.
        """
        untapped_lands = [land for land in self.battlefield if land.is_land() and not land.tapped]

        while not card.is_playable(self.mana_pool) and untapped_lands:
            land = untapped_lands.pop(0)
            possible_mana = land.get_possible_mana()
            chosen_mana = self.choose_mana_to_produce(possible_mana)
            mana_produced = land.produce_mana(chosen_mana)
            if mana_produced:
                life_cost = land.life_cost_to_activate()
                if life_cost > 0:
                    self.health -= life_cost
                    logger.info(f"{self.name} pays {life_cost} life to activate {land.name}.")
                    logger.info(f"{self.name} is now at {self.health} health.")

                land.tap()

                for color, amount in mana_produced.items():
                    self.mana_pool[color] = self.mana_pool.get(color, 0) + amount
                logger.info(f"{self.name} adds {amount} {color} mana to their mana pool by tapping {land.name}.")
                logger.info(f"Mana pool after tapping {land.name}: {self.mana_pool}")
            else:
                logger.warning(f"{self.name} could not produce mana with {land.name}.")

    def declare_attackers(self, players: List['Player']) -> List['Card']:
        attackers = []
        opponent = self.get_opponent(players)

        for card in self.battlefield:
            if card.is_creature() and not card.tapped and not card.summoning_sick:
                if opponent.health <= self.get_creature_power(card):
                    attackers.append(card)
                    card.tap()
                else:
                    potential_blockers = [c for c in opponent.battlefield if c.is_creature() and not c.tapped]
                    if not potential_blockers:
                        attackers.append(card)
                        card.tap()
                    else:
                        weakest_blocker_toughness = min(self.get_creature_toughness(c) for c in potential_blockers)
                        if self.get_creature_power(card) > weakest_blocker_toughness:
                            attackers.append(card)
                            card.tap()

        if attackers:
            logger.info(f"{self.name} declares attackers: {[card.name for card in attackers]}")
        else:
            logger.info(f"{self.name} has no creatures to attack with.")
        return attackers

    def declare_blockers(self, attackers: List['Card']) -> Dict['Card', Optional['Card']]:
        blockers: Dict['Card', Optional['Card']] = {}
        available_blockers = [card for card in self.battlefield if card.is_creature() and not card.tapped]

        for attacker in attackers:
            if available_blockers:
                blocker = available_blockers.pop(0)
                blocker.tap()
                logger.info(f"{self.name} blocks {attacker.name} with {blocker.name}.")
                blockers[attacker] = blocker
            else:
                blockers[attacker] = None
        return blockers

# --- MODIFY THE GAME ENGINE ---
class GameEngine:
    """
    Manages the overall flow of the game.
    """
    def __init__(self, players: List[Player]):
        if len(players) < 2:
            raise ValueError("A minimum of two players is required to start the game.")
        self.players = players
        self.game_stack = Stack()
        self.rules_engine = RulesEngine()
        self.current_player_index = 0
        self.attackers: List['Card'] = []
        self.blockers: Dict['Card', Optional['Card']] = {}
        self.game_over = False

        # Set game_stack for players
        for player in players:
            player.game_stack = self.game_stack

    def initialize_game(self) -> None:
        """
        Initializes the game by having each player draw their initial hands.
        """
        logger.info("Initializing the game...")
        for player in self.players:
            for _ in range(7):
                player.draw_card()
            logger.info(f"{player.name} has drawn their initial hand.")

    def is_game_over(self) -> bool:
        """
        Checks if the game is over by verifying the status of all players.
        """
        for player in self.players:
            if player.has_lost():
                return True
        return False

    def declare_winner(self):
        for player in self.players:
            if not player.has_lost():
                logger.info(f"{player.name} wins the game!")
                return
        logger.info("The game ended in a draw.")

    def log_player_state(self, current_player: Player, opponent: Player) -> None:
        """
        Logs the current state of the player and the opponent.
        """
        logger.info(f"Current player state: {current_player}")
        logger.info(f"Opponent state: {opponent}")
        logger.info(f"Current player's battlefield: {[card.name for card in current_player.battlefield]}")
        logger.info(f"Opponent's battlefield: {[card.name for card in opponent.battlefield]}")
        logger.info(f"Current player's hand: {[card.name for card in current_player.hand]}")
        logger.info(f"Opponent's hand: {[card.name for card in opponent.hand]}")

    def take_turn(self):
        """
        Executes a single turn for the current player, including actions and rewards.
        """
        if self.is_game_over():
            return

        current_player = self.players[self.current_player_index]
        opponent = self.players[(self.current_player_index + 1) % len(self.players)]
        logger.info(f"\n=== Starting turn for {current_player.name} ===")
        self.log_player_state(current_player, opponent)

        # Reset the phase to Untap at the beginning of each turn
        self.rules_engine.to_Untap()

        # Update previous state before starting the turn
        self.update_player_previous_state(current_player)
        self.update_player_previous_state(opponent)

        # Go through all phases
        while True:
            phase = self.rules_engine.state
            logger.info(f"Phase: {phase}")
            self.execute_phase_actions(current_player)

            if self.is_game_over():
                self.game_over = True
                break

            if phase == "Cleanup":
                break

            self.rules_engine.next_phase()

            self.update_player_previous_state(current_player)
            self.update_player_previous_state(opponent)

        logger.info(f"=== Ending turn for {current_player.name} ===")

        # Advance to the next player
        self.current_player_index = (self.current_player_index + 1) % len(self.players)

    def update_player_previous_state(self, player: Player):
        player.previous_health = player.health
        player.previous_creature_count = len([card for card in player.battlefield if card.is_creature()])
        player.previous_board_state_value = player.evaluate_board_state()
        player.previous_battlefield = player.battlefield.copy()
        player.previous_lands_played = player.lands_played_this_turn

    def execute_phase_actions(self, player: Player) -> None:
        """
        Executes actions allowed in the current phase for the player.
        """
        phase = self.rules_engine.state
        opponent = self.players[(self.current_player_index + 1) % len(self.players)]
        logger.info(f"Executing {phase} phase for {player.name}")

        # Reset took_action flag
        player.took_action = False

        if phase == "Untap" and self.rules_engine.is_action_allowed("untap_all"):
            player.untap_all()

        elif phase == "Upkeep":
            logger.info(f"{player.name}'s upkeep phase.")

        elif phase == "Draw" and self.rules_engine.is_action_allowed("draw_card"):
            player.draw_card()

        elif phase in ["Main1", "Main2"]:
            if isinstance(player, AIPlayer):
                # AI player logic
                player.play_lands()
                player.agent.take_action_and_calculate_reward(player, opponent, self.game_stack)
            else:
                logger.info(f"{player.name} is making decisions in {phase}.")
                # Placeholder for human player actions

            self.game_stack.resolve()

        elif phase == "BeginCombat":
            logger.info(f"{player.name}'s begin combat phase.")

        elif phase == "DeclareAttackers" and self.rules_engine.is_action_allowed("declare_attackers"):
            self.attackers = player.declare_attackers(self.players)

        elif phase == "DeclareBlockers" and self.rules_engine.is_action_allowed("declare_blockers"):
            self.blockers = opponent.declare_blockers(self.attackers)

        elif phase == "CombatDamage" and self.rules_engine.is_action_allowed("resolve_combat"):
            player.resolve_combat(self.blockers, opponent)

        elif phase == "EndCombat":
            logger.info(f"{player.name}'s end combat phase.")

        elif phase == "End":
            logger.info(f"{player.name}'s end phase.")

        elif phase == "Cleanup" and self.rules_engine.is_action_allowed("discard_down_to_max_hand_size"):
            player.discard_down_to_max_hand_size()
            player.empty_mana_pool()

        logger.info(f"End of {phase} phase for {player.name}")

# --- RULES ENGINE USING FINITE STATE MACHINE ---
class RulesEngine:
    """
    Manages the phases of the game using a finite state machine.
    """
    def __init__(self):
        self.states = [
            "Untap", "Upkeep", "Draw", "Main1", "BeginCombat", "DeclareAttackers",
            "DeclareBlockers", "CombatDamage", "EndCombat", "Main2", "End", "Cleanup"
        ]
        self.machine = Machine(model=self, states=self.states, initial="Untap")

        # Define phase transitions
        self.machine.add_transition(trigger="next_phase", source="Untap", dest="Upkeep")
        self.machine.add_transition(trigger="next_phase", source="Upkeep", dest="Draw")
        self.machine.add_transition(trigger="next_phase", source="Draw", dest="Main1")
        self.machine.add_transition(trigger="next_phase", source="Main1", dest="BeginCombat")
        self.machine.add_transition(trigger="next_phase", source="BeginCombat", dest="DeclareAttackers")
        self.machine.add_transition(trigger="next_phase", source="DeclareAttackers", dest="DeclareBlockers")
        self.machine.add_transition(trigger="next_phase", source="DeclareBlockers", dest="CombatDamage")
        self.machine.add_transition(trigger="next_phase", source="CombatDamage", dest="EndCombat")
        self.machine.add_transition(trigger="next_phase", source="EndCombat", dest="Main2")
        self.machine.add_transition(trigger="next_phase", source="Main2", dest="End")
        self.machine.add_transition(trigger="next_phase", source="End", dest="Cleanup")
        self.machine.add_transition(trigger="next_phase", source="Cleanup", dest="Untap")

    def is_action_allowed(self, action: str) -> bool:
        """
        Determines if the specified action is allowed in the current phase.
        """
        allowed_actions = {
            "Untap": ["untap_all"],
            "Draw": ["draw_card"],
            "DeclareAttackers": ["declare_attackers"],
            "DeclareBlockers": ["declare_blockers"],
            "CombatDamage": ["resolve_combat"],
            "Cleanup": ["discard_down_to_max_hand_size"],
        }
        
        current_phase = self.state
        if current_phase in allowed_actions:
            return action in allowed_actions[current_phase]
        return False




# --- PYGAME DISPLAY ---
class PygameDisplay:
    def __init__(self, width: int, height: int):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Magic: The Gathering")
        self.font = pygame.font.SysFont(None, 25)  # Adjust font size
        self.clock = pygame.time.Clock()
        self.running = True

    def draw_field(self, player1: Player, player2: Player):
        """
        Draw the game field with two battlefields and a line in the middle.
        """
        self.screen.fill((0, 128, 0))  # Green background for game board
        pygame.draw.line(self.screen, (255, 255, 255), (0, 300), (800, 300), 2)  # Line across the middle

        # Display each player's battlefield and hand
        self._draw_player_cards(player1, (50, 350))
        self._draw_player_cards(player2, (50, 50))

        # Display player names and health
        self._draw_text(f"{player1.name}: {player1.health} HP", (10, 550))
        self._draw_text(f"{player2.name}: {player2.health} HP", (10, 10))

    def _draw_player_cards(self, player: Player, start_pos):
        x, y = start_pos
        card_width, card_height = 100, 150
        # Group cards
        lands = [card for card in player.battlefield if card.is_land()]
        creatures = [card for card in player.battlefield if card.is_creature()]
        others = [card for card in player.battlefield if not card.is_land() and not card.is_creature()]

        # Draw lands
        self._draw_card_group(lands, x, y, card_width, card_height, "Lands")

        # Draw creatures
        y += card_height + 30
        self._draw_card_group(creatures, x, y, card_width, card_height, "Creatures")

        # Draw other permanents
        y += card_height + 30
        self._draw_card_group(others, x, y, card_width, card_height, "Others")

    def _draw_card_group(self, cards, x, y, card_width, card_height, label):
        self._draw_text(label, (x, y - 20))
        for i, card in enumerate(cards):
            pos_x = x + i * (card_width + 10)
            if card.image:
                scaled_image = pygame.transform.scale(card.image, (card_width, card_height))
                self.screen.blit(scaled_image, (pos_x, y))
            else:
                self._draw_card_placeholder(pos_x, y, card_width, card_height)
            self._draw_text(card.name, (pos_x, y + card_height + 5))

    def _draw_card_placeholder(self, x, y, width, height):
        pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(x, y, width, height))
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(x, y, width, height), 2)

    def _draw_text(self, text, position):
        """
        Renders text on the screen at the given position.
        """
        label = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(label, position)

    def run_game_loop(self, player1: Player, player2: Player, game_engine: GameEngine):
        """
        Runs the game loop to keep Pygame running.
        """
        while self.running and not game_engine.is_game_over():
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False

            # Call game engine to handle logic and process a turn
            game_engine.take_turn()

            # Redraw the field after each turn
            self.draw_field(player1, player2)
            pygame.display.flip()
            self.clock.tick(1)  # Control the speed of the game

        # Announce winner
        game_engine.declare_winner()
        pygame.quit()

import pandas as pd
import pyarrow.parquet as pq

# Load cards from Parquet file using PyArrow
def load_cards_from_parquet(parquet_path: str) -> List[Card]:
    """
    Loads cards from a Parquet file and returns a list of Card instances.
    """
    try:
        # Load the Parquet file with pyarrow
        table = pq.read_table(parquet_path)
        df = table.to_pandas()  # Convert the Arrow Table to a Pandas DataFrame
        
        cards = []
        for _, row in df.iterrows():
            try:
                card = Card.from_parquet(row)
                # Exclude certain card types that should not be in decks
                if 'Vanguard' not in card.card_type and 'Plane' not in card.card_type:
                    cards.append(card)
            except Exception as e:
                logger.warning(f"Skipping card due to error: {e}")

        if not cards:
            logger.error("No valid cards were loaded from the Parquet file.")
        else:
            logger.info(f"Loaded {len(cards)} cards from Parquet.")
        return cards

    except FileNotFoundError:
        logger.error(f"Parquet file not found: {parquet_path}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading cards: {e}")
        return []

import random

def build_deck(card_pool, land_pool, colors=['W'], min_land=20, max_land=24, deck_size=60):
    """
    Builds a deck with a random number of land cards between min_land and max_land.
    The rest of the deck is filled with non-land cards from the card pool.
    The deck will include cards and lands of the specified colors.
    """
    # Filter the card pool to include only cards of the specified colors
    filtered_card_pool = [card for card in card_pool if set(card.colors).intersection(colors)]

    # Further filter for low mana cost cards (mana value <= 3)
    low_cost_cards = [card for card in filtered_card_pool if card.calculate_mana_value() <= 3]

    # If not enough low-cost cards, use all filtered cards
    if len(low_cost_cards) < deck_size - max_land:
        non_land_cards = filtered_card_pool
    else:
        non_land_cards = low_cost_cards

    # Filter the land pool to include only lands that produce the specified colors
    def land_produces_colors(land, colors):
        possible_mana = land.get_possible_mana()
        return any(color in possible_mana for color in colors)

    filtered_land_pool = [land for land in land_pool if land_produces_colors(land, colors)]

    if not non_land_cards or not filtered_land_pool:
        raise ValueError("Not enough cards or lands of the specified colors to build the deck.")

    # Ensure randomness in land count within min-max bounds
    num_land_cards = random.randint(min_land, max_land)

    # Select land cards
    if len(filtered_land_pool) < num_land_cards:
        # If not enough unique lands, allow duplicates
        land_cards = random.choices(filtered_land_pool, k=num_land_cards)
    else:
        land_cards = random.sample(filtered_land_pool, num_land_cards)

    # Select non-land cards
    num_non_land_cards = deck_size - num_land_cards
    if len(non_land_cards) < num_non_land_cards:
        # If not enough unique cards, allow duplicates
        non_land_cards = random.choices(non_land_cards, k=num_non_land_cards)
    else:
        non_land_cards = random.sample(non_land_cards, num_non_land_cards)

    # Combine land and non-land cards into the deck
    deck = land_cards + non_land_cards
    random.shuffle(deck)  # Shuffle the deck

    return deck


# --- STACK CLASSES ---
class StackItem:
    def __init__(self, player: Player, card: Card):
        self.player = player
        self.card = card

    def resolve(self):
        # Implement resolution logic here
        if 'Instant' in self.card.card_type or 'Sorcery' in self.card.card_type:
            # Apply the effect of the instant or sorcery
            logger.info(f"{self.player.name} resolves {self.card.name}.")
            # Placeholder: Implement actual effect logic based on oracle_text
            self.player.move_to_graveyard(self.card)
        else:
            # For permanents, move them to the battlefield
            self.player.battlefield.append(self.card)
            logger.info(f"{self.card.name} has been resolved and is now on the battlefield.")


class Stack:
    def __init__(self):
        self.items = []

    def add_to_stack(self, item: StackItem):
        self.items.append(item)
        logger.info(f"{item.card.name} added to the stack.")

    def resolve(self):
        while self.items:
            item = self.items.pop()
            logger.info(f"Resolving {item.card.name} from the stack.")
            item.resolve()


# --- MAIN SCRIPT ---
import pandas as pd

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    # Load cards from Parquet
    data_path = "/home/plunder/data/mtgdata.parquet"
    cards = load_cards_from_parquet(data_path)

    # Separate land and non-land cards
    land_pool = [card for card in cards if card.is_land()]
    card_pool = [card for card in cards if not card.is_land()]

    if land_pool and len(card_pool) >= 100:
        # Define number of games to train
        num_games_to_train = 1000  # You can change this value or make it infinite

        # Initialize metrics tracking
        metrics = {
            "game_num": [],
            "bot1_wins": 0,
            "bot2_wins": 0,
            "bot1_total_damage": [],
            "bot2_total_damage": [],
            "bot1_cards_played": [],
            "bot2_cards_played": [],
            "bot1_avg_reward": [],
            "bot2_avg_reward": []
        }

        for game_num in range(num_games_to_train):
            logger.info(f"Starting Game {game_num + 1}/{num_games_to_train}")

            # Build decks for both players with specified colors
            deck1_cards = build_deck(card_pool, land_pool, colors=['W'], min_land=20, max_land=24, deck_size=60)
            deck2_cards = build_deck(card_pool, land_pool, colors=['G'], min_land=20, max_land=24, deck_size=60)

            deck1 = Deck(cards=deck1_cards)
            deck2 = Deck(cards=deck2_cards)

            # Initialize AI agents
            state_size = 9  # Corrected based on actual features
            latent_dim = 128
            max_hand_size = 7
            action_size = max_hand_size + 1

            agent1 = Agent(state_size, action_size, latent_dim)
            agent2 = Agent(state_size, action_size, latent_dim)

            player1 = AIPlayer(name="Bot1", deck=deck1, agent=agent1)
            player2 = AIPlayer(name="Bot2", deck=deck2, agent=agent2)

            # Create game engine
            game = GameEngine(players=[player1, player2])
            game.initialize_game()

            # Run game simulation
            game_display = PygameDisplay(1024, 768)
            game_display.run_game_loop(player1, player2, game)

            # After the game, update metrics
            winner = None
            if player1.has_lost():
                metrics["bot2_wins"] += 1
                winner = "Bot2"
            elif player2.has_lost():
                metrics["bot1_wins"] += 1
                winner = "Bot1"

            # Track damage dealt and cards played
            bot1_damage = player2.previous_health - player2.health
            bot2_damage = player1.previous_health - player1.health

            metrics["bot1_total_damage"].append(bot1_damage)
            metrics["bot2_total_damage"].append(bot2_damage)

            metrics["bot1_cards_played"].append(len(player1.battlefield))
            metrics["bot2_cards_played"].append(len(player2.battlefield))

            # Calculate and track average rewards
            bot1_avg_reward = sum([exp[3] for exp in agent1.replay_buffer]) / len(agent1.replay_buffer)
            bot2_avg_reward = sum([exp[3] for exp in agent2.replay_buffer]) / len(agent2.replay_buffer)

            metrics["bot1_avg_reward"].append(bot1_avg_reward)
            metrics["bot2_avg_reward"].append(bot2_avg_reward)

            # Log the game results
            logger.info(f"Game {game_num + 1} complete. Winner: {winner}")
            logger.info(f"Bot1 - Damage Dealt: {bot1_damage}, Cards Played: {len(player1.battlefield)}, Avg Reward: {bot1_avg_reward}")
            logger.info(f"Bot2 - Damage Dealt: {bot2_damage}, Cards Played: {len(player2.battlefield)}, Avg Reward: {bot2_avg_reward}")

            # Save the AI models after each game
            torch.save(agent1.q_network.state_dict(), f"mtg_ai_model_bot1_game_{game_num + 1}.pth")
            torch.save(agent2.q_network.state_dict(), f"mtg_ai_model_bot2_game_{game_num + 1}.pth")

        # At the end, save metrics to a CSV file
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv("mtg_ai_training_metrics.csv", index=False)
        logger.info("Training complete. Metrics saved to mtg_ai_training_metrics.csv.")
    else:
        logger.error("Insufficient cards available to start the game.")
