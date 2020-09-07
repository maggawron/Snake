# import keyboard
import Screen
import Backend
import pygame


def main():
    pygame.init()
    game_width = 25
    game_length = 25
    clock = pygame.time.Clock()
    stan = Backend.State(game_width, game_length)
    gridsize = 20

    screen = pygame.display.set_mode((game_width * gridsize, game_length * gridsize), 0, 32)

    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    Screen.drawGrid(surface)

    myfont = pygame.font.SysFont("monospace", 16)

    snake_dead = False
    while not snake_dead:
        clock.tick(3)
        key = Screen.handle_keys()
        #TODO
        if key is None:
            key = stan.prev_key
        Screen.drawGrid(surface)
        stan.move_snake(key)
        snake_dead = not stan.check_if_snake_lives()
        Screen.draw(stan, surface)
        screen.blit(surface, (0, 0))
        text = myfont.render("Score {0}".format(stan.points), 1, (0, 0, 0))
        screen.blit(text, (5, 10))
        pygame.display.update()

"""
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
"""
if __name__ == "__main__":
    main()
