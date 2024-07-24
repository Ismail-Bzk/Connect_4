import numpy as np
import pygame
import random
from math import sqrt, log
import math
import copy
import sys

# Initialization of Pygame
pygame.init()

# Define Colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
LIGHT_RED = (255, 200, 200)
LIGHT_GREEN = (200, 255, 200)

# Get screen resolution
infoObject = pygame.display.Info()
screen_width = infoObject.current_w
screen_height = infoObject.current_h

# Define Dimensions
ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARESIZE = 100

# Calculate maximum possible SQUARESIZE based on screen size
max_square_width = screen_width // COLUMN_COUNT
max_square_height = screen_height // (ROW_COUNT + 2)
SQUARESIZE = min(SQUARESIZE, max_square_width, max_square_height)

RADIUS = int(SQUARESIZE / 2 - 5)
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 2) * SQUARESIZE
size = (width, height)

# Configure Window
screen = pygame.display.set_mode(size)
myfont = pygame.font.SysFont("monospace", 75)
smallfont = pygame.font.SysFont("monospace", 25)
signature_font = pygame.font.SysFont("monospace", 20,bold=True)


class ConnectFourBoard:
    def __init__(self):
        self.board = self.create_board()
        self.turn = 1  # Start with Player 1

    def create_board(self):
        return np.zeros((ROW_COUNT, COLUMN_COUNT))

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def is_valid_location(self, col):
        return self.board[ROW_COUNT - 1][col] == 0

    def get_next_open_row(self, col):
        for r in range(ROW_COUNT):
            if self.board[r][col] == 0:
                return r
        return None

    def print_board(self):
        print(np.flip(self.board, 0))

    def winning_move(self, piece):
        # Check horizontal locations
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if (self.board[r][c] == piece and self.board[r][c + 1] == piece and
                        self.board[r][c + 2] == piece and self.board[r][c + 3] == piece):
                    return True

        # Check vertical locations
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if (self.board[r][c] == piece and self.board[r + 1][c] == piece and
                        self.board[r + 2][c] == piece and self.board[r + 3][c] == piece):
                    return True

        # Check positively sloped diagonals
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if (self.board[r][c] == piece and self.board[r + 1][c + 1] == piece and
                        self.board[r + 2][c + 2] == piece and self.board[r + 3][c + 3] == piece):
                    return True

        # Check negatively sloped diagonals
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if (self.board[r][c] == piece and self.board[r - 1][c + 1] == piece and
                        self.board[r - 2][c + 2] == piece and self.board[r - 3][c + 3] == piece):
                    return True
        return False

    def legal_moves(self):
        return [col for col in range(COLUMN_COUNT) if self.is_valid_location(col)]

    def terminal(self):
        return self.winning_move(1) or self.winning_move(2) or len(self.legal_moves()) == 0

    def score(self):
        if self.winning_move(1):
            return 1.0
        elif self.winning_move(2):
            return 0.0
        else:
            return 0.5

    def play(self, col, piece):
        row = self.get_next_open_row(col)
        if row is not None:
            self.drop_piece(row, col, piece)

    def playout(self):
        current_player = self.turn
        while not self.terminal():
            legal_moves = self.legal_moves()
            move = random.choice(legal_moves)
            self.play(move, current_player)
            current_player = 3 - current_player  # Switch player
        return self.score()

    def draw_board(self):
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(screen, BLACK, (int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                if self.board[r][c] == 1:
                    pygame.draw.circle(screen, RED, (int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2) - SQUARESIZE), RADIUS)
                elif self.board[r][c] == 2:
                    pygame.draw.circle(screen, YELLOW, (int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2) - SQUARESIZE), RADIUS)
        pygame.display.update()

# Standard UCB algorithm
def ucb(board, n):
    moves = board.legal_moves()
    sum_scores = [0.0 for _ in range(len(moves))]
    nb_visits = [0 for _ in range(len(moves))]

    for i in range(n):
        best_move = 0
        best_ucb = float('-inf')

        for m in range(len(moves)):
            if nb_visits[m] == 0:
                score = float('inf')
            else:
                score = sum_scores[m] / nb_visits[m] + 0.4 * sqrt(log(i + 1) / nb_visits[m])
            if score > best_ucb:
                best_ucb = score
                best_move = m

        b = copy.deepcopy(board)
        b.play(moves[best_move], board.turn)
        r = b.playout()

        if board.turn == 2:  # If it's the AI's turn (Player 2)
            r = 1 - r

        sum_scores[best_move] += r
        nb_visits[best_move] += 1

        # Determine the state-specific explanation
        explanation_text = f"Iteration {i + 1}: Evaluating column {moves[best_move] + 1}\nVisits: {nb_visits[best_move]}\nScore: {sum_scores[best_move]:.2f}"
        
        if board.winning_move(2):
            explanation_text += "\nAI found a potential winning move."
            highlight_color = LIGHT_GREEN
        elif board.winning_move(1):
            explanation_text += "\nAI is blocking the opponent's winning move."
            highlight_color = LIGHT_RED
        elif sum_scores[best_move] > 0.7:
            explanation_text += "\nAI considers this a strong move for future advantage."
            highlight_color = LIGHT_GREEN
        elif sum_scores[best_move] < 0.3:
            explanation_text += "\nAI is avoiding this weak move."
            highlight_color = LIGHT_RED
        else:
            explanation_text += "\nAI is exploring potential future outcomes."
            highlight_color = WHITE

        # Visualize the iteration process less frequently
        if i % 100 == 0:  # Update visualization every 100 iterations
            screen.fill(WHITE, (0, 0, width, SQUARESIZE))  # Clear previous text
            lines = explanation_text.split('\n')
            for index, line in enumerate(lines):
                explanation = smallfont.render(line, 1, BLACK)
                screen.blit(explanation, (10, 10 + index * 20))

            # Draw line to the evaluated piece
            row = board.get_next_open_row(moves[best_move])
            pygame.draw.line(screen, GREEN, (int(moves[best_move] * SQUARESIZE + SQUARESIZE / 2), SQUARESIZE),
                             (int(moves[best_move] * SQUARESIZE + SQUARESIZE / 2), (ROW_COUNT - row) * SQUARESIZE + SQUARESIZE // 2), 2)
            pygame.draw.circle(screen, highlight_color, (int(moves[best_move] * SQUARESIZE + SQUARESIZE / 2), int(SQUARESIZE / 2)), RADIUS)
            pygame.display.update()
            pygame.time.wait(50)
            board.draw_board()

    best_score = float('-inf')
    best_move = 0

    for m in range(len(moves)):
        if nb_visits[m] > best_score:
            best_score = nb_visits[m]
            best_move = m

    return moves[best_move]

# Enhanced RAVE algorithm
def rave(board, n):
    moves = board.legal_moves()
    sum_scores = [0.0 for _ in range(len(moves))]
    nb_visits = [0 for _ in range(len(moves))]
    amaf_sum_scores = [0.0 for _ in range(COLUMN_COUNT)]  # Initialize for all columns
    amaf_nb_visits = [0 for _ in range(COLUMN_COUNT)]  # Initialize for all columns

    for i in range(n):
        best_move = 0
        best_ucb = float('-inf')

        for m in range(len(moves)):
            if nb_visits[m] == 0:
                score = float('inf')
            else:
                beta = amaf_nb_visits[moves[m]] / (amaf_nb_visits[moves[m]] + nb_visits[m] + 1e-6)  # avoid division by zero
                if amaf_nb_visits[moves[m]] > 0:  # Only compute the score if amaf_nb_visits[m] is greater than 0
                    score = (1 - beta) * (sum_scores[m] / nb_visits[m]) + beta * (amaf_sum_scores[moves[m]] / amaf_nb_visits[moves[m]])
                    score += 0.4 * sqrt(log(i + 1) / nb_visits[m])
                else:
                    score = (sum_scores[m] / nb_visits[m]) + 0.4 * sqrt(log(i + 1) / nb_visits[m])
            if score > best_ucb:
                best_ucb = score
                best_move = m

        b = copy.deepcopy(board)
        b.play(moves[best_move], board.turn)
        r = b.playout()

        if board.turn == 2:  # If it's the AI's turn (Player 2)
            r = 1 - r

        sum_scores[best_move] += r
        nb_visits[best_move] += 1
        for move in moves:
            amaf_sum_scores[move] += r
            amaf_nb_visits[move] += 1

        # Determine the state-specific explanation
        explanation_text = f"Iteration {i + 1}: Evaluating column {moves[best_move] + 1}\nVisits: {nb_visits[best_move]}\nScore: {sum_scores[best_move]:.2f}"
        
        if board.winning_move(2):
            explanation_text += "\nAI found a potential winning move."
            highlight_color = LIGHT_GREEN
        elif board.winning_move(1):
            explanation_text += "\nAI is blocking the opponent's winning move."
            highlight_color = LIGHT_RED
        elif sum_scores[best_move] > 0.7:
            explanation_text += "\nAI considers this a strong move for future advantage."
            highlight_color = LIGHT_GREEN
        elif sum_scores[best_move] < 0.3:
            explanation_text += "\nAI is avoiding this weak move."
            highlight_color = LIGHT_RED
        else:
            explanation_text += "\nAI is exploring potential future outcomes."
            highlight_color = WHITE

        # Visualize the iteration process less frequently
        if i % 100 == 0:  # Update visualization every 100 iterations
            screen.fill(WHITE, (0, 0, width, SQUARESIZE))  # Clear previous text
            lines = explanation_text.split('\n')
            for index, line in enumerate(lines):
                explanation = smallfont.render(line, 1, BLACK)
                screen.blit(explanation, (10, 10 + index * 20))

            # Draw line to the evaluated piece
            row = board.get_next_open_row(moves[best_move])
            pygame.draw.line(screen, GREEN, (int(moves[best_move] * SQUARESIZE + SQUARESIZE / 2), SQUARESIZE),
                             (int(moves[best_move] * SQUARESIZE + SQUARESIZE / 2), (ROW_COUNT - row) * SQUARESIZE + SQUARESIZE // 2), 2)
            pygame.draw.circle(screen, highlight_color, (int(moves[best_move] * SQUARESIZE + SQUARESIZE / 2), int(SQUARESIZE / 2)), RADIUS)
            pygame.display.update()
            pygame.time.wait(50)
            board.draw_board()

    best_score = float('-inf')
    best_move = 0

    for m in range(len(moves)):
        if nb_visits[m] > best_score:
            best_score = nb_visits[m]
            best_move = m

    return moves[best_move]


def puct(board, n, c_puct=1.0):
    moves = board.legal_moves()
    sum_scores = [0.0 for _ in range(len(moves))]
    nb_visits = [0 for _ in range(len(moves))]
    prior_prob = [1 / len(moves) for _ in range(len(moves))]  # Assuming uniform prior probabilities

    for i in range(n):
        best_move = 0
        best_puct = float('-inf')

        for m in range(len(moves)):
            if nb_visits[m] == 0:
                score = float('inf')
            else:
                p = prior_prob[m]
                u = c_puct * p * sqrt(sum(nb_visits)) / (1 + nb_visits[m])
                score = sum_scores[m] / nb_visits[m] + u
            if score > best_puct:
                best_puct = score
                best_move = m

        b = copy.deepcopy(board)
        b.play(moves[best_move], board.turn)
        r = b.playout()

        if board.turn == 2:  # If it's the AI's turn (Player 2)
            r = 1 - r

        sum_scores[best_move] += r
        nb_visits[best_move] += 1

        # Determine the state-specific explanation
        explanation_text = f"Iteration {i + 1}: Evaluating column {moves[best_move] + 1}\nVisits: {nb_visits[best_move]}\nScore: {sum_scores[best_move]:.2f}"
        
        if board.winning_move(2):
            explanation_text += "\nAI found a potential winning move."
            highlight_color = LIGHT_GREEN
        elif board.winning_move(1):
            explanation_text += "\nAI is blocking the opponent's winning move."
            highlight_color = LIGHT_RED
        elif sum_scores[best_move] > 0.7:
            explanation_text += "\nAI considers this a strong move for future advantage."
            highlight_color = LIGHT_GREEN
        elif sum_scores[best_move] < 0.3:
            explanation_text += "\nAI is avoiding this weak move."
            highlight_color = LIGHT_RED
        else:
            explanation_text += "\nAI is exploring potential future outcomes."
            highlight_color = WHITE

        # Visualize the iteration process less frequently
        if i % 100 == 0:  # Update visualization every 100 iterations
            screen.fill(WHITE, (0, 0, width, SQUARESIZE))  # Clear previous text
            lines = explanation_text.split('\n')
            for index, line in enumerate(lines):
                explanation = smallfont.render(line, 1, BLACK)
                screen.blit(explanation, (10, 10 + index * 20))

            # Draw line to the evaluated piece
            row = board.get_next_open_row(moves[best_move])
            pygame.draw.line(screen, GREEN, (int(moves[best_move] * SQUARESIZE + SQUARESIZE / 2), SQUARESIZE),
                             (int(moves[best_move] * SQUARESIZE + SQUARESIZE / 2), (ROW_COUNT - row) * SQUARESIZE + SQUARESIZE // 2), 2)
            pygame.draw.circle(screen, highlight_color, (int(moves[best_move] * SQUARESIZE + SQUARESIZE / 2), int(SQUARESIZE / 2)), RADIUS)
            pygame.display.update()
            pygame.time.wait(50)
            board.draw_board()

    best_score = float('-inf')
    best_move = 0

    for m in range(len(moves)):
        if nb_visits[m] > best_score:
            best_score = nb_visits[m]
            best_move = m

    return moves[best_move]

def nested_mc_search(board, level):
    if level == 0 or board.terminal():
        return board.playout(), None

    best_score = -float('inf')
    best_move = None

    for move in board.legal_moves():
        new_board = copy.deepcopy(board)
        new_board.play(move, new_board.turn)
        score, _ = nested_mc_search(new_board, level - 1)

        if new_board.turn == 2:  # Si c'est le tour de l'IA (Joueur 2)
            score = 1 - score

        if score > best_score:
            best_score = score
            best_move = move

    return best_score, best_move

def nmcs(board, level=3):
    _, move = nested_mc_search(board, level)
    return move



def multiplayer_game():
    board = ConnectFourBoard()
    board.draw_board()
    game_over = False
    turn = 0  # Start with Player 1 (Human)

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, WHITE, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                if turn == 0:
                    pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)
                else:
                    pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE / 2)), RADIUS)
                pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, WHITE, (0, 0, width, SQUARESIZE))
                if turn == 0:
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))

                    if board.is_valid_location(col):
                        row = board.get_next_open_row(col)
                        board.play(col, 1)

                        if board.winning_move(1):
                            label = myfont.render("Player 1 wins!!", 1, RED)
                            screen.blit(label, (40, 10))
                            game_over = True

                        turn = 1
                        board.draw_board()

                else:
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))

                    if board.is_valid_location(col):
                        row = board.get_next_open_row(col)
                        board.play(col, 2)

                        if board.winning_move(2):
                            label = myfont.render("Player 2 wins!!", 1, YELLOW)
                            screen.blit(label, (40, 10))
                            game_over = True

                        turn = 0
                        board.draw_board()

        if game_over:
            pygame.time.wait(3000)

def play_with_ai(algorithm):
    board = ConnectFourBoard()
    board.draw_board()
    game_over = False
    turn = 0  # Start with Player 1 (Human)

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if turn == 0 and event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, WHITE, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                col = int(math.floor(posx / SQUARESIZE))

                if board.is_valid_location(col):
                    row = board.get_next_open_row(col)
                    board.play(col, 1)

                    if board.winning_move(1):
                        label = myfont.render("Player 1 wins!!", 1, RED)
                        screen.blit(label, (40, 10))
                        game_over = True

                    turn = 1
                    board.draw_board()

        if not game_over and turn == 1:
            screen.fill(WHITE, (0, 0, width, SQUARESIZE))  # Clear previous text
            explanation = smallfont.render(f"AI ({algorithm.upper()}) is thinking...", 1, BLACK)
            screen.blit(explanation, (10, 10))
            pygame.display.update()

            if algorithm == 'ucb':
                col = ucb(board, 1000)
            elif algorithm == 'rave':
                col = rave(board, 1000)
            elif algorithm == 'puct':
                col = puct(board, 1000)
            elif algorithm == 'nmcs':
                col = nmcs(board, level=3)

            screen.fill(WHITE, (0, 0, width, SQUARESIZE))  # Clear the explanation text

            if board.is_valid_location(col):
                row = board.get_next_open_row(col)
                board.play(col, 2)

                if board.winning_move(2):
                    label = myfont.render(f"AI ({algorithm.upper()}) wins!!", 1, YELLOW)
                    screen.blit(label, (40, 10))
                    game_over = True

                turn = 0
                board.draw_board()

        if game_over:
            pygame.time.wait(3000)

def ai_vs_ai():
    board = ConnectFourBoard()
    board.draw_board()
    game_over = False
    turn = 0  # Start with Player 1 (AI - UCB)
    algorithm1 = "ucb"
    algorithm2 = "rave"
    
    while not game_over:
        if turn == 0:
            screen.fill(WHITE, (0, 0, width, SQUARESIZE))  # Clear previous text
            explanation = smallfont.render("AI 1 (UCB) is thinking...", 1, BLACK)
            screen.blit(explanation, (10, 10))
            pygame.display.update()
            col = ucb(board, 1000)  # AI 1 move with UCB
            screen.fill(WHITE, (0, 0, width, SQUARESIZE))  # Clear the explanation text

            if board.is_valid_location(col):
                row = board.get_next_open_row(col)
                board.play(col, 1)
                board.print_board()  # Print board after AI's move

                if board.winning_move(1):
                    label = myfont.render("AI 1 (UCB) wins!!", 1, RED)
                    screen.blit(label, (40, 10))
                    game_over = True

                turn = 1
                board.draw_board()

        if not game_over and turn == 1:
            screen.fill(WHITE, (0, 0, width, SQUARESIZE))  # Clear previous text
            explanation = smallfont.render("AI 2 (RAVE) is thinking...", 1, BLACK)
            screen.blit(explanation, (10, 10))
            pygame.display.update()
            col = rave(board, 1000)  # AI 2 move with RAVE
            screen.fill(WHITE, (0, 0, width, SQUARESIZE))  # Clear the explanation text

            if board.is_valid_location(col):
                row = board.get_next_open_row(col)
                board.play(col, 2)
                board.print_board()  # Print board after AI's move

                if board.winning_move(2):
                    label = myfont.render("AI 2 (RAVE) wins!!", 1, YELLOW)
                    screen.blit(label, (40, 10))
                    game_over = True

                turn = 0
                board.draw_board()

        if game_over:
            pygame.time.wait(3000)

def main_menu():
    while True:
        
        screen.fill(WHITE)
        label = myfont.render("Connect Four", 1, BLACK)
        screen.blit(label, (40, 10))
        
        multiplayer_button = smallfont.render("1. Multiplayer", 1, BLACK)
        screen.blit(multiplayer_button, (40, 150))

        ucb_button = smallfont.render("2. Play against UCB", 1, BLACK)
        screen.blit(ucb_button, (40, 200))

        rave_button = smallfont.render("3. Play against RAVE", 1, BLACK)
        screen.blit(rave_button, (40, 250))
        
        puct_button = smallfont.render("4. Play against PUCT", 1, BLACK)
        screen.blit(puct_button, (40, 300))

        nmcs_button = smallfont.render("5. Play against NMCS", 1, BLACK)
        screen.blit(nmcs_button, (40, 350))

        ai_vs_ai_button = smallfont.render("6. UCB vs RAVE", 1, BLACK)
        screen.blit(ai_vs_ai_button, (40, 400))
        
        
        signature = signature_font.render("By BOUZKKA Ismail & FADIL Anas",
                                          1,
                                          BLACK)
        screen.blit(signature, (width - 380, height -80))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    multiplayer_game()
                elif event.key == pygame.K_2:
                    play_with_ai('ucb')
                elif event.key == pygame.K_3:
                    play_with_ai('rave')
                elif event.key == pygame.K_4:
                    play_with_ai('puct')
                elif event.key == pygame.K_5:
                    play_with_ai('nmcs')
                elif event.key == pygame.K_6:
                    ai_vs_ai()


def test_c_puct_values(board, n_iterations, c_puct_values):
    results = {}
    for c_puct in c_puct_values:
        wins = 0
        for _ in range(n_iterations):
            result = puct(board, 1000, c_puct=c_puct)
            if result == 1:  # Assuming 1 represents a win for the AI
                wins += 1
        win_rate = wins / n_iterations
        results[c_puct] = win_rate
        print(f"c_puct = {c_puct}, win rate = {win_rate}")
    return results


if __name__ == "__main__":
    main_menu()
    
