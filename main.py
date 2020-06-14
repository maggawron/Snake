import keyboard
from Screen import Screen
from Backend import Backend
#from timer import Timer

#def pressed(event):
#    print(event.name)

def main():
    #t = Timer()

    #setup game parameters
    display = Screen()
    b = Backend()
    width = 50
    length = 10
    prev_key = None
    apple_loc = []
    obs_number = 3
    apple_number = 2
    obstacle_loc = b.generate_obstacle(width, length, obs_number)
    snake_loc = b.generate_first_snake(obstacle_loc, width, length)
    #print(snake_loc)
    for _ in range(apple_number):
        apple_loc = b.generate_apple(snake_loc, obstacle_loc, apple_loc, width, length)

    #initiate intro screen of the game
    display.print_screen(width, length, obstacle_loc, apple_loc, snake_loc)

    #keyboard.on_press(pressed)

    while b.check_if_snake_lives(snake_loc, obstacle_loc, width, length):
        key = keyboard.read_key()
        snake_loc = b.update_snake(key, prev_key, snake_loc, apple_loc)
        apple_eaten = 0

        for apple in apple_loc:
            if b.check_if_apple_eaten(snake_loc, apple_loc):
                apple_loc.remove(apple)
                apple_loc = b.generate_apple(snake_loc, obstacle_loc, apple_loc, width, length)

        b.cls()
        display.print_screen(width, length, obstacle_loc, apple_loc, snake_loc)

# check
if __name__ == "__main__":
    main()