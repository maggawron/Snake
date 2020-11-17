import os
import sys
import pygame
from Screen import Screen
from State import State


def convert_env_to_state(stan, all_screens):

    for screen in all_screens:
        #encoding = {[1, 0, 0, 0, 0]: "apple",
        #            [0, 1, 0, 0, 0]: "obstacle",
        #            [0, 0, 1, 0, 0]: "head",
        #            [0, 0, 0, 1, 0]: "body",
        #            [0, 0, 0, 0, 1]: "empty"}
        body_loc = []
        obstacle_loc = []
        apple_loc = []
        head = []
        for r in range(stan.game_length):
            for c in range(stan.game_width):
                assert len(screen[r][c]) == 5
                if screen[r][c] == [1, 0, 0, 0, 0]:
                    apple_loc.append((r, c))
                elif screen[r][c] == [0, 1, 0, 0, 0]:
                    obstacle_loc.append((r, c))
                elif screen[r][c] == [0, 0, 1, 0, 0]:
                    head.append((r, c))
                    #print("found head")
                elif screen[r][c] == [0, 0, 0, 1, 0]:
                    body_loc.append((r, c))
                elif screen[r][c] == [0, 0, 0, 0, 1]:
                    pass
                else:
                    print(screen[r][c])

        #print(head, body_loc)
        snake_loc = body_loc + head
        stan.snake_loc = snake_loc
        stan.obstacle_loc = obstacle_loc
        stan.apple_loc = apple_loc

def display_evaluation_game(input_file):
    pygame.init()
    stan = State()
    game_screen = Screen(stan.game_width, stan.game_length)
    clock = pygame.time.Clock()

    with open(input_file, "r") as fp:
        for line in fp:
            single_move = eval(line)
            screen, reward_print, total_reward = single_move
            clock.tick(5)
            convert_env_to_state(stan, screen)
            game_screen.display_move(stan)
            text2 = game_screen.myfont.render("Score: {:0.0f} | Step reward: {:0.2f} | Total reward: {:0.1f}"
                                              .format(stan.points, reward_print, total_reward), 1, (0, 0, 0))
            game_screen.game_display.blit(text2, (10, 10))
            pygame.display.update()


def main():
    if len(sys.argv) <= 2:
        raise ValueError(f"Usage: ./{sys.argv[0]} experiment_name [start end]")
    experiment_name = sys.argv[1]
    start = 1
    end = 10
    if len(sys.argv) == 4:
        start = int(sys.argv[2])
        end = int(sys.argv[3])
    for i in range(start, end):
        filepath = os.path.join("saved_models", experiment_name,
                                "eval_data", f"Eval_data.step{i}.txt")
        print("Step: ", i)
        display_evaluation_game(filepath)


if __name__ == "__main__":
    main()
