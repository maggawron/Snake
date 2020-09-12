'''Use pygame for snake development'''
import random
import copy


# random.seed(42)

def _random_generator(width, length):
    los_row = random.randrange(2, length - 1)
    los_col = random.randrange(2, width - 1)
    return los_row, los_col


def generate_obstacle(width, length, obs_number):
    obstacle_loc = []
    for _ in range(obs_number):
        los_row, los_col = _random_generator(width, length)
        obstacle_loc.append((los_row, los_col))
    return obstacle_loc


def generate_first_snake(obstacle_loc, width, length):
    los_row, los_col = _random_generator(width, length)
    while (los_row, los_col) in obstacle_loc:
        los_row, los_col = _random_generator(width, length)
    return [(los_row, los_col)]


def generate_apple(snake_loc, obstacle_loc, apple_loc, width, length):
    """If snake ate an apple, generate him another one, but not on obstacle, border, other apple or snake"""
    snake_loc_set = set(snake_loc)
    obstacle_loc_set = set(obstacle_loc)
    apple_loc_set = set(apple_loc)
    set_of_forbidden = apple_loc_set | obstacle_loc_set | snake_loc_set
    los_row, los_col = _random_generator(width, length)
    while (los_row, los_col) in set_of_forbidden:
        los_row, los_col = _random_generator(width, length)
    apple_loc.append((los_row, los_col))
    return apple_loc


class State:
    def __init__(self, width, length):
        self.level = 1
        self.game_width = width
        self.game_length = length
        obs_number = 0
        self.prev_key = "right"
        self.obstacle_loc = generate_obstacle(self.game_width, self.game_length, obs_number)
        #self.snake_loc = generate_first_snake(self.obstacle_loc, self.game_width, self.game_length)
        self.snake_loc = [(12, 13)] #TODO snake begins in the middle of the field
        apple_number = 50 #TODO
        self.apple_loc = []
        for _ in range(apple_number):
            self.apple_loc = generate_apple(self.snake_loc, self.obstacle_loc, self.apple_loc, self.game_width, self.game_length)
        self.points = 0
        self.ate_apple = False
        assert self.check_if_snake_lives()

    def check_if_snake_lives(self):
        """Return False if snake crashed into obstacle, itself or border"""
        head_row, head_col = self.snake_loc[-1]
        if head_row == 0 or head_row == self.game_length - 1:
            return False
        if head_col == 0 or head_col == self.game_width - 1:
            return False

        all_obstacles = set(self.snake_loc[:-1]) | set(self.obstacle_loc)
        if (head_row, head_col) in all_obstacles:
            return False
        return True

    def move_snake(self, key):
        """Move snake to the new position"""
        apple_loc_set = set(self.apple_loc)

        opposite = {"left": "right",
                    "right": "left",
                    "up": "down",
                    "down": "up"}
        assert key in opposite, print(key)
        # Keep same direction for invalid move order.
        if opposite[key] == self.prev_key:
            key = self.prev_key

        row, col = self.snake_loc[-1]
        if key == "up":
            new_row, new_col = row - 1, col
        elif key == "down":
            new_row, new_col = row + 1, col
        elif key == "left":
            new_row, new_col = row, col - 1
        elif key == "right":
            new_row, new_col = row, col + 1
        self.snake_loc.append((new_row, new_col))

        # Keep last element if snake ate an apple
        if (new_row, new_col) not in apple_loc_set:
            #TODO fix O(n) with deque?
            self.snake_loc.pop(0)

        self.ate_apple = False
        # Check if snake ate an apple
        for apple in self.apple_loc:
            if self.snake_loc[-1] == apple:
                #random.seed(42)
                self.apple_loc.remove(apple)
                self.apple_loc = generate_apple(
                    self.snake_loc, self.obstacle_loc, self.apple_loc, self.game_width, self.game_length)
                self.points += self.level
                self.level = self.update_level(self.points, self.level)
                self.ate_apple = True
                break
        self.prev_key = key

    def update_level(self, points, level):
        if points % 12 == 0:
            level += 1
        return level

stable_state = State(25, 25) #
state_const = copy.deepcopy(stable_state)

