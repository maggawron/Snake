import state
import time
import keyboard
import Screen
import Backend

def pressed_callback(event, stan):
    """Called on every key press"""
    with stan.lock:
        key = event.name
        stan.was_pressed = True
        stan.move_snake(key)

def main():
    width = 50
    length = 10
    stan = Backend.State(width, length)

    #initiate intro screen of the game
    Screen.print_screen(width, length, stan.obstacle_loc, stan.apple_loc, stan.snake_loc)
    keyboard.on_press(lambda event: pressed_callback(event, stan))

    while stan.check_if_snake_lives():
        time.sleep(0.5 / stan.level)
        Screen.print_screen(stan.width, stan.length, stan.obstacle_loc, stan.apple_loc, stan.snake_loc)
        print("Number of points:", stan.points, " | Level:", stan.level)
        with stan.lock:
            if stan.was_pressed:
                stan.was_pressed = False
                continue
            stan.move_snake(stan.prev_key)
    stan.snake_dead = True
    print(f"Game over | Points: {stan.points} | Level: {stan.level}")

if __name__ == "__main__":
    main()




