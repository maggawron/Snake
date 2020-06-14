'''Use pygame for snake development'''
import random
import os

class Backend:
    # Get a previous location of a snake and return the next one
    def update_snake(self, key, prev_key, snake_loc, apple_loc):
        apple_loc_set = set(apple_loc)
        if key is None:
            # keep moving in the previous direction
            pass
        else:
            row, col = snake_loc[-1]
            # snake_loc.pop(0)
            if prev_key != "down" and key == "up":
                new_row = row - 1
                new_col = col
            if prev_key != "up" and key == "down":
                new_row = row + 1
                new_col = col
            if prev_key != "right" and key == "left":
                new_row = row
                new_col = col - 1
            if prev_key != "left" and key == "right":
                new_row = row
                new_col = col + 1
            snake_loc.append((new_col, new_row))
        if (new_row, new_col) not in apple_loc_set:
            # Duplicate last location so that when next move happens, snake is same length
            snake_loc.pop(0)
        return snake_loc

    # check if snake ate an apple and generate a new one in case old was eaten
    def check_if_apple_eaten(self, snake_loc, apple_loc):
        # Check if snake ate an apple
        for apple in apple_loc:
            if snake_loc[-1] == apple:
                return True

    #function to return True if snake crashed into obstacle, itself or border
    def check_if_snake_lives(self, snake_loc, obstacle_loc, width, length):
        head_row, head_col = snake_loc[-1]
        #print("snake head", head_row, head_col)
        if head_row == 0 or head_row == length - 1:
            return False
        if head_col == 0 or head_col == width - 1:
            return False

        all_obstacles = set(snake_loc[:-1]) | set(obstacle_loc)
        if (head_row, head_col) in all_obstacles:
            return False
        return True

    def generate_obstacle(self, width, length, obs_number):
        obstacle_loc = []
        for _ in range(obs_number):
            los_row, los_col = self._random_generator(width, length)
            obstacle_loc.append((los_row, los_col))
        return obstacle_loc

    def generate_first_snake(self, obstacle_loc, width, length):
        los_row, los_col = self._random_generator(width, length)
        while (los_row, los_col) in obstacle_loc:
            los_row, los_col = self._random_generator(width, length)
        return [(los_row, los_col)]

    def generate_apple(self, snake_loc, obstacle_loc, apple_loc, width, length):
        # If snake ate an apple, generate him another one, but not on obstacle, border, other apple or snake
        snake_loc_set = set(snake_loc)
        obstacle_loc_set = set(obstacle_loc)
        #print("apple loc before set creation", apple_loc)
        apple_loc_set = set(apple_loc)
        set_of_forbidden = apple_loc_set | obstacle_loc_set | snake_loc_set
        #print("all", set_of_forbidden)
        los_row, los_col = self._random_generator(width, length)
        while (los_row, los_col) in set_of_forbidden:
            los_row, los_col = self._random_generator(width, length)
        apple_loc.append((los_row, los_col))
        #print("new apple loc", apple_loc)
        return apple_loc

    def _random_generator(self, width, length):
        los_row = random.randrange(2, length - 1)
        los_col = random.randrange(2, width - 1)
        return los_row, los_col

    def cls(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def tests(self):
        pass







