# import keyboard
from Screen import Screen
from State import State
import pygame
import sys


def handle_keys():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                return "up"
            elif event.key == pygame.K_DOWN:
                return "down"
            elif event.key == pygame.K_LEFT:
                return "left"
            elif event.key == pygame.K_RIGHT:
                return "right"
    return None


def main():
    pygame.init() #Initialize pygame
    stan = State() #Initialize state of the game
    game_screen = Screen(stan.game_width, stan.game_length) #Initialize game screen
    snake_dead = False
    clock = pygame.time.Clock()


    while not snake_dead:
        clock.tick(5)
        key = handle_keys()
        if key is None:
            key = stan.prev_key
        stan.move_snake(key)
        snake_dead = not stan.check_if_snake_lives()
        text = game_screen.myfont.render("Score {0}".format(stan.points), 1, (0, 0, 0))
        game_screen.game_display.blit(text, (5, 10))
        game_screen.display_move(stan) #Update game screen
        pygame.display.update()


if __name__ == "__main__":
    main()
