import keyboard
import Screen
import Backend
import time
from threading import Lock
import numpy as np

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

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

    def move_snake(self, key):
        """Move snake to the new position"""
        self.snake_loc = Backend.update_snake(key, self.prev_key, self.snake_loc, self.apple_loc)
        self.points -= 1

        for apple in self.apple_loc:
            if Backend.check_if_apple_eaten(self.snake_loc, apple):
                self.apple_loc.remove(apple)
                self.apple_loc = Backend.generate_apple(
                    self.snake_loc, self.obstacle_loc, self.apple_loc, self.width, self.length)
                self.points += self.level * 10
                self.level = self.update_level(self.points, self.level)

        Backend.cls()
        Screen.print_screen(self.width, self.length, self.obstacle_loc, self.apple_loc, self.snake_loc)
        print("Number of points:", self.points, " | level:", self.level)
        self.prev_key = key

    def update_level(self, points, level):
        if points % 12 == 0:
            level += 1
        return level

def pressed_callback(event, state):
    """Called on every key press"""
    with state.lock:
        key = event.name
        state.was_pressed = True
        state.move_snake(key)

def main():
    width = 50
    length = 10
    state = State(width, length)

    #initiate intro screen of the game
    Screen.print_screen(width, length, state.obstacle_loc, state.apple_loc, state.snake_loc)
    keyboard.on_press(lambda event: pressed_callback(event, state))

    while Backend.check_if_snake_lives(state.snake_loc, state.obstacle_loc, width, length):
        time.sleep(0.5 / state.level)
        with state.lock:
            if state.was_pressed:
                state.was_pressed = False
                continue
            state.move_snake(state.prev_key)
    state.points -= 30
    print(f"Game over | Points: {state.points}, Level: {state.level}")

if __name__ == "__main__":
    main()




