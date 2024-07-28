# AI for Connect Four using Monte Carlo Tree Search (MCTS) with UCB1, RAVE, PUCT, and NRPA

This project implements the Connect Four game using advanced Monte Carlo Tree Search (MCTS) algorithms to create a high-performing AI. The algorithms used include UCB1, RAVE, PUCT, and NRPA.

## Introduction

This project leverages enhanced Monte Carlo Tree Search (MCTS) with different variants to make decisions by simulating game outcomes and using statistical analysis to choose the best move. The implemented algorithms are:

- UCB1 (Upper Confidence Bounds for Trees)
- RAVE (Rapid Action Value Estimation)
- PUCT (Polynomial Upper Confidence Trees)
- NRPA (Nested Rollout Policy Adaptation)

## Concepts Used

### Connect Four Game

Connect Four is a two-player board game where players alternately drop colored discs into a vertical grid. The objective is to connect four of one's discs in a row, either horizontally, vertically, or diagonally, before the opponent.

### Monte Carlo Tree Search (MCTS)

MCTS is a search algorithm used for decision-making processes, particularly in games. It has four main steps:
1. **Selection**: Starting from the root, select a child node until reaching a leaf node.
2. **Expansion**: Expand the leaf node by adding one or more child nodes.
3. **Simulation**: Simulate a random rollout from the new node to a terminal state.
4. **Backpropagation**: Propagate the result of the simulation back through the tree to update the nodes.

### UCB (Upper Confidence Bounds for Trees)

Uses the UCB1 formula to balance exploration and exploitation during move selection.

### RAVE (Rapid Action Value Estimation)

Combines action values from previous simulations to accelerate the algorithm's convergence.

### PUCT (Polynomial Upper Confidence Trees)

Integrates value and policy predictions provided by a predictive model to guide the search tree exploration.

### NRPA (Nested Rollout Policy Adaptation)

Dynamically adapts the rollout policy during the search to improve efficiency and results.

## Code Implementation

### Libraries and Initialization

We use the following libraries:
- `numpy` for board management.
- `pygame` for graphical representation and user interaction.
- `math` and `random` for calculations and random simulations.

### Colors and Dimensions

We define colors and dimensions for the game board and user interface elements.

### `ConnectFourBoard` Class

This class encapsulates the game logic and board management:
- `create_board`: Initializes an empty game board.
- `drop_piece`: Inserts a piece into a specific column.
- `is_valid_location`: Checks if a column can receive a piece.
- `get_next_open_row`: Finds the next available row in a column.
- `print_board`: Displays the current state of the board.
- `winning_move`: Checks if a move is a winning move.
- `legal_moves`: Returns a list of legal moves (valid columns).
- `terminal`: Checks if the game is over.
- `score`: Calculates the score of the game.
- `play`: Plays a move on the board.
- `playout`: Simulates a random game from the current state to a terminal state.
- `draw_board`: Draws the current state of the board using Pygame.

### Implementation of `ucb`

The `ucb` function uses the UCB algorithm to select moves. Here's how it is implemented:
1. **Initialization**: Initialize scores and visits for each legal move.
2. **Selection**: For each simulation, select the move with the highest UCB value.
3. **Simulation**: Perform a simulation (playout) for the selected move.
4. **Update**: Update scores and visits based on the simulation result.

### Implementation of `rave`

The `rave` function uses the RAVE algorithm to select moves. Here's how it is implemented:
1. Calculate the values of \(Q(s, a)\) and \(\tilde{Q}(s, a)\) for each action.
2. Calculate the weight \(\beta(s, a)\) using the given formulas.
3. Update scores and visits after each simulation.
4. Select the action with the best \(Q_*(s, a)\) value.

### Implementation of `puct`

The `puct` function uses the PUCT algorithm to select moves. Here's how it is implemented:
1. **Initialization**: Initialize scores and visits for each legal move.
2. **Value Calculation**: Calculate the values of \(U(s, a)\) for each move by combining predictive probabilities and UCB values.
3. **Selection**: For each simulation, select the move with the highest \(U(s, a)\) value.
4. **Simulation**: Perform a simulation (playout) for the selected move.
5. **Update**: Update scores and visits based on the simulation result.

### Implementation of `nrpa`

The `nrpa` function uses the NRPA algorithm to select moves. Here's how it is implemented:
1. **Policy Adaptation**: Uses gradient ascent to adjust the policy based on successful rollout sequences.
2. **Rollout**: Performs random rollouts according to the current policy and records sequences.
3. **Recursion**: For each search level, recursively calls the lower level with the updated policy.
4. **Update**: Adjusts the policy based on the best-found sequences.

## Game Modes

The program offers several game modes:
- Multiplayer: Two human players compete against each other.
- Human vs AI: A human player competes against an AI using one of the implemented algorithms (UCB, RAVE, PUCT, NRPA).
- AI vs AI: Two AIs using one of the implemented algorithms play against each other.

## Conclusion

This implementation demonstrates the use of enhanced Monte Carlo Tree Search (MCTS) methods to create an AI that plays Connect Four. By simulating game outcomes and analyzing statistical data, the algorithm makes informed decisions to choose the best move.

## Authors

- Ismail Bouzkka
