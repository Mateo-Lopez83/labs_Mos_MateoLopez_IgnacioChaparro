"""
Queen Mapper

Description:
This program visualizes the attack paths of queens on a chess board.
The user can specify the positions of the queens and the program will
highlight the squares that are under attack. The program also checks
if all squares are covered by the queens' attack paths and displays a
message accordingly.

Author: Juan Andrés Méndez.

"""

import pygame

# Constants and Configuration BOARD_SIZE = 8
SQUARE_SIZE = 64
BOARD_SIZE = 8
WINDOW_SIZE = SQUARE_SIZE * (BOARD_SIZE + 1)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
LABEL_OFFSET = SQUARE_SIZE // 2

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE), pygame.RESIZABLE)
pygame.display.flip()
pygame.display.set_caption("Queen Mapper")
font = pygame.font.Font(None, 36)


class ChessBoard:
    """Class to represent a chess board and its pieces."""

    def __init__(self):
        self.queen_image = pygame.image.load("queen.png").convert_alpha()
        self.queen_image = pygame.transform.scale(
            self.queen_image, (SQUARE_SIZE, SQUARE_SIZE)
        )

    def draw_board(self, highlight_squares):
        """Draw the chess board and highlight attack squares."""
        screen.fill(COLOR_BLACK)
        for i in range(BOARD_SIZE):
            self.render_text(
                str(8 - i),
                (5, LABEL_OFFSET + i * SQUARE_SIZE + SQUARE_SIZE // 2),
                COLOR_WHITE,
            )
            self.render_text(
                chr(i + 97),
                (LABEL_OFFSET + i * SQUARE_SIZE + SQUARE_SIZE // 2, 5),
                COLOR_WHITE,
            )

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                rect = (
                    LABEL_OFFSET + col * SQUARE_SIZE,
                    LABEL_OFFSET + row * SQUARE_SIZE,
                    SQUARE_SIZE,
                    SQUARE_SIZE,
                )
                pygame.draw.rect(
                    screen, COLOR_WHITE if (row + col) % 2 == 0 else COLOR_BLACK, rect
                )
                if (row, col) in highlight_squares:
                    self.highlight_square(rect, COLOR_RED)

    def place_queen(self, position):
        """Place a queen on the board."""
        screen.blit(
            self.queen_image,
            (
                LABEL_OFFSET + position[1] * SQUARE_SIZE,
                LABEL_OFFSET + position[0] * SQUARE_SIZE,
            ),
        )

    def highlight_square(self, rect, color):
        """Highlight a square on the board."""
        surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
        surface.set_alpha(75)
        surface.fill(color)
        screen.blit(surface, rect)

    @staticmethod
    def notation_to_position(notation):
        """Convert chess notation to board position."""
        letter_to_number = dict(zip("abcdefgh", range(8)))
        col = letter_to_number[notation[0]]
        row = 8 - int(notation[1])
        return row, col

    @staticmethod
    def calculate_attack_squares(position):
        """Calculate all squares a queen can attack from a given position."""
        row, col = position
        attack_squares = {(row, i) for i in range(BOARD_SIZE)} | {
            (i, col) for i in range(BOARD_SIZE)
        }
        attack_squares |= {
            (row + i, col + i)
            for i in range(-BOARD_SIZE, BOARD_SIZE)
            if 0 <= row + i < BOARD_SIZE and 0 <= col + i < BOARD_SIZE
        }
        attack_squares |= {
            (row + i, col - i)
            for i in range(-BOARD_SIZE, BOARD_SIZE)
            if 0 <= row + i < BOARD_SIZE and 0 <= col - i < BOARD_SIZE
        }
        return attack_squares

    def render_text(self, text, position, color):
        """Render text on the screen."""
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, position)


def visualize_queens(positions):
    """Main function to run the program."""
    chess_board = ChessBoard()
    attack_squares = set()

    for pos in positions:
        position = ChessBoard.notation_to_position(pos)
        attack_squares.update(ChessBoard.calculate_attack_squares(position))

    chess_board.draw_board(attack_squares)

    for pos in positions:
        position = ChessBoard.notation_to_position(pos)
        chess_board.place_queen(position)

    message = (
        "All squares are covered!"
        if len(attack_squares) == BOARD_SIZE**2
        else "Not all squares are covered!"
    )
    message_color = COLOR_GREEN if len(attack_squares) == BOARD_SIZE**2 else COLOR_RED
    chess_board.render_text(message, (5, WINDOW_SIZE - 40), message_color)

    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return


if __name__ == "__main__":
    queen_positions = ["a1", "g5", "f7", "b2","e4","d7"]
    visualize_queens(queen_positions)
