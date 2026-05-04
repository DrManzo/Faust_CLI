import random

# Initialize the game board as a 10x10 grid of zeros
board = [[0] * 10 for _ in range(10)]

# Place the 1x5 ship horizontally starting at (0, 4)
ship_x, ship_y = 0, 4
for j in range(5):
    board[ship_x][ship_y + j] = 'S'


# Function to print the game board
def print_board():
    print('  ', end='')
    for i in range(10):
        print(i, end=' ')
    print()
    for i in range(10):
        print(i, end=' ')
        for j in range(10):
            cell = board[i][j]
            if cell == 'S':
                # Hide ship and show empty water
                print(' ', end=' ')
            elif cell == 1:
                print('X', end=' ')
            else:
                print(' ', end=' ')
        print()


# Main game loop
while True:
    # Ask the player for their move (x, y coordinates)
    x = int(input("Enter x coordinate: "))
    y = int(input("Enter y coordinate: "))

    # Check if the move is within the board boundaries
    if x < 0 or x >= 10 or y < 0 or y >= 10:
        print("Invalid move! Try again.")
        continue

    # Check if the move hits the ship
    if board[x][y] == 'S':
        print("Hit!")
        board[x][y] = 1
    else:
        print("Miss!")

    # Print the updated game board
    print_board()

    # Check if the player has won (all parts of the ship sunk)
    if all(cell != 'S' for row in board for cell in row):
        print("Congratulations! You've won!")
        break