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
            clock.tick(1)
            convert_env_to_state(stan, screen)
            game_screen.display_move(stan)
            text2 = game_screen.myfont.render("Score: {:0.0f} | Step reward: {:0.2f} | Total reward: {:0.1f}"
                                              .format(stan.points, reward_print, total_reward), 1, (0, 0, 0))
            game_screen.game_display.blit(text2, (10, 10))
            pygame.display.update()


def main():
    for i in range(1520, 1521):
        filepath = rf"C:\Users\ibm\PycharmProjects\Snake\eval_data\Eval_data.step{i}.txt"
        #filepath = f"eval_data/Eval_data.step{i}.txt"
        print("Step: ", i)
        display_evaluation_game(filepath)


if __name__ == "__main__":
    main()
