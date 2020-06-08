'''Use pygame for snake development'''
import random

class Backend:

    # Get a previous location of a snake and return the next one
    def update_snake(self, key, prev_key, snake_loc, apple_loc):
        if snake_loc[-1] == apple_loc:
            # Duplicate last location so that when next move happens, snake is same length
            snake_loc.intert(0, apple_loc)
        if key is None:
            # keep moving in the previous direction
            continue
        else:
            snake_loc.pop(0)
            row, col = snake_loc(-1)
            if prev_key != "down" and key == "up":
                snake_loc.append(row, col + 1)
            if prev_key != "up" and key == "down":
                snake_loc.append(row, col - 1)
            if prev_key != "right" and key == "left":
                snake_loc.append(row - 1, col)
            if prev_key != "left" and key == "right":
                snake_loc.append(row + 1, col)
        return snake_loc

    # check if snake ate an apple and generate a new one in case old was eaten
    def update_apple(self, snake_loc, obstacle_loc, apple_loc, width, length):
        apple_eaten = 0

        # Check if snake ate an apple
        for apple in apple_loc:
            if snake_loc(-1) == apple:
                apple_eaten = 1
                apple_loc.remove(apple)

        # If snake ate an apple, generate him another one, but not on obstacle, border, other apple or snake
        if apple_eaten == 1:
            set_of_forbidden = set(snake_loc + obstacle_loc + apple_loc)
            los_row = random.randrange(2, width - 1)
            los_col = random.randrange(2, length - 1)
            while (los_row, los_col) in set_of_forbidden:
                los_row = random.randrange(2, width - 1)
                los_col = random.randrange(2, length - 1)
            apple_loc.append(los_row, los_col)
        return apple_loc

    def check_if_snake_crashed(self):

        pass

    def generate_obstacle(self):
        pass

    def generate_first_snake(self):
        pass

    def generate_first_apples(self):
        pass

    def tests(self):







