import pygame
from Screen import Screen
from State import State


def display_evaluation_game(input_file):
    pygame.init()
    stan = State()
    game_screen = Screen(stan.game_width, stan.game_length)
    clock = pygame.time.Clock()

    with open(input_file, "r") as fp:
        for line in fp:
            single_move = eval(line)
            stan.snake_loc, stan.apple_loc, stan.obstacle_loc, reward_print, total_reward, stan.points = single_move
            clock.tick(5)
            game_screen.display_move(stan)
            text2 = game_screen.myfont.render("Score: {:0.0f} | Step reward: {:0.2f} | Total reward: {:0.1f}"
                                              .format(stan.points, reward_print, total_reward), 1, (0, 0, 0))
            game_screen.game_display.blit(text2, (10, 10))
            pygame.display.update()


def main():
    for i in range(1, 5):
        filepath = rf"C:\Users\ibm\PycharmProjects\Snake\eval_data\Eval_data.step{i}.txt"
        #filepath = f"eval_data/Eval_data.step{i}.txt"
        print("Step: ", i)
        display_evaluation_game(filepath)


if __name__ == "__main__":
    main()
