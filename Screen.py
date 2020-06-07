import random
class Screen:
    # Frame of a game
    def print_screen(self, width, length, obstacle_loc, apple_loc, snake_loc):
        border_sign = "$"
        obstacle_sign = "X"
        apple_sign = "O"
        snake_sign = "S"

        for row_id in range(length):
            for col_id in range(width):
                if row_id == 0 or col_id == 0 or row_id == length - 1 or col_id == width - 1:
                    print(border_sign, end="")
                elif (row_id, col_id) in obstacle_loc:
                    print(obstacle_sign, end="")
                elif (row_id, col_id) in apple_loc:
                    print(apple_sign, end="")
                elif (row_id, col_id) in snake_loc:
                    print(snake_sign, end="")
                else:
                    print(" ", end="")
                #End a line
                if col_id == width - 1:
                    print("")

display = Screen()
width = 50
length = 10
obstacle_loc = {(2,7), (4,5)}
apple_loc = {(7,9)}
snake_loc = {(5,6), (6,6), (7,6), (8,6), (8,7), (8,8)}

display.print_screen(width, length, obstacle_loc, apple_loc, snake_loc)

