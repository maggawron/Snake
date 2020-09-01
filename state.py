import Screen
import Backend
from threading import Lock

# TODO: Probably need to modify main in manual game to use generate_game method to initialize game to cut double code
class State:
    def __init__(self, width, length):
        self.level = 1
        self.width = width
        self.length = length
        obs_number = 3
        self.prev_key = "right"
        self.obstacle_loc = Backend.generate_obstacle(width, length, obs_number)
        self.snake_loc = Backend.generate_first_snake(self.obstacle_loc, width, length)
        apple_number = 2
        self.apple_loc = []
        for _ in range(apple_number):
            self.apple_loc = Backend.generate_apple(self.snake_loc, self.obstacle_loc, self.apple_loc, width, length)
        self.was_pressed = False
        self.lock = Lock()
        self.points = 0
        self.snake_dead = False
        self.ate_apple = False

    def generate_game(self):
        obs_number = 3
        self.prev_key = "right"
        self.obstacle_loc = Backend.generate_obstacle(self.width, self.length, obs_number)
        self.snake_loc = Backend.generate_first_snake(self.obstacle_loc, self.width, self.length)
        apple_number = 2
        self.apple_loc = []
        for _ in range(apple_number):
            self.apple_loc = Backend.generate_apple(self.snake_loc, self.obstacle_loc, self.apple_loc, self.width, self.length)
        self.was_pressed = False
        self.lock = Lock()
        self.points = 0
        self.snake_dead = False
        self.ate_apple = False

    def move_snake(self, key):
        """Move snake to the new position"""
        self.snake_loc = Backend.update_snake(key, self.prev_key, self.snake_loc, self.apple_loc)

        for apple in self.apple_loc:
            if Backend.check_if_apple_eaten(self.snake_loc, apple):
                self.apple_loc.remove(apple)
                self.apple_loc = Backend.generate_apple(
                    self.snake_loc, self.obstacle_loc, self.apple_loc, self.width, self.length)
                self.points += self.level
                self.level = self.update_level(self.points, self.level)
                self.ate_apple = True
            else:
                self.ate_apple = False

        self.prev_key = key

    def update_level(self, points, level):
        if points % 12 == 0:
            level += 1
        return level

