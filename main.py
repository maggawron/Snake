import keyboard
from Screen import Screen
from Backend import Backend
from time import time
import time

class State:
    def __init__(self, width, length):
        self.width = width
        self.length = length
        obs_number = 3
        self.prev_key = "right"
        b = Backend()
        self.obstacle_loc = b.generate_obstacle(width, length, obs_number)
        self.snake_loc = b.generate_first_snake(self.obstacle_loc, width, length)
        apple_number = 2
        self.apple_loc = []
        for _ in range(apple_number):
            self.apple_loc = b.generate_apple(self.snake_loc, self.obstacle_loc, self.apple_loc, width, length)
        self.was_pressed = False

def move_snake(key, state):
    b = Backend()
    state.snake_loc = b.update_snake(key, state.prev_key, state.snake_loc, state.apple_loc)

    for apple in state.apple_loc:
        if b.check_if_apple_eaten(state.snake_loc, state.apple_loc):
            state.apple_loc.remove(apple)
            state.apple_loc = b.generate_apple(state.snake_loc, state.obstacle_loc, state.apple_loc, state.width,
                                               state.length)
    b.cls()
    state.prev_key = key
    display = Screen()
    display.print_screen(state.width, state.length, state.obstacle_loc, state.apple_loc, state.snake_loc)

def pressed_callback(event, state):
    key = event.name
    state.was_pressed = True
    move_snake(key, state)

def main():
    #setup game parameters
    display = Screen()
    b = Backend()
    prev_key = None
    width = 50
    length = 10
    state = State(width, length)

    #initiate intro screen of the game
    display.print_screen(width, length, state.obstacle_loc, state.apple_loc, state.snake_loc)
    keyboard.on_press(lambda event: pressed_callback(event, state))

    while b.check_if_snake_lives(state.snake_loc, state.obstacle_loc, width, length):
        time.sleep(0.5)
        if state.was_pressed:
            state.was_pressed = False
            continue
        move_snake(state.prev_key, state)

# check
if __name__ == "__main__":
    main()

