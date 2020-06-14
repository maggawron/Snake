def print_screen(width, length, obstacle_loc, apple_loc, snake_loc):
    """Print current state of the game"""
    border_sign = "$"
    obstacle_sign = "X"
    apple_sign = "O"
    snake_sign = "S"

    snake_loc_set = set(snake_loc)
    obstacle_loc_set = set(obstacle_loc)
    apple_loc_set = set(apple_loc)

    #Print screen line by line
    for row_id in range(length):
        for col_id in range(width):
            if row_id == 0 or col_id == 0 or row_id == length - 1 or col_id == width - 1:
                print(border_sign, end="")
            elif (row_id, col_id) in snake_loc_set:
                print(snake_sign, end="")
            elif (row_id, col_id) in obstacle_loc_set:
                print(obstacle_sign, end="")
            elif (row_id, col_id) in apple_loc_set:
                print(apple_sign, end="")
            else:
                print(" ", end="")
            #End a line
            if col_id == width - 1:
                print("")
