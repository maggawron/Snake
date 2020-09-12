import pygame
import Backend
import sys

gridsize = 20
snake_color = (17, 24, 47)
head_color = (255, 1, 1)
apple_color = (223, 163, 49)
obstacle_color = (150, 150, 150)
game_width = 25
game_length = 25


def print_all(input_file):
    pygame.init()
    game_width = 25
    game_length = 25
    clock = pygame.time.Clock()
    stan = Backend.State(game_width, game_length)
    gridsize = 20
    screen = pygame.display.set_mode((game_width * gridsize, game_length * gridsize), 0, 32)
    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    drawGrid(surface)
    myfont = pygame.font.SysFont("monospace", 16)

    with open(input_file, "r") as fp:
        for line in fp:
            single_move = eval(line)
            stan.snake_loc, stan.apple_loc, stan.obstacle_loc = single_move
            print_move(stan, clock, surface, screen, myfont)


def print_move(stan, clock, surface, screen, myfont):
    clock.tick(10)
    drawGrid(surface)
    draw(stan, surface)
    screen.blit(surface, (0, 0))
    text = myfont.render("Score {0}".format(len(stan.snake_loc[:-1])*10), 1, (0, 0, 0))
    screen.blit(text, (5, 10))
    pygame.display.update()


def draw_object(surface, obj, color):
    r = pygame.Rect((obj[1] * gridsize, obj[0] * gridsize), (gridsize, gridsize))
    pygame.draw.rect(surface, color, r)
    pygame.draw.rect(surface, (93, 216, 228), r, 1)


def draw(state, surface):
    for p in state.snake_loc[:-1]:
        draw_object(surface, p, snake_color)
    head = state.snake_loc[-1]
    draw_object(surface, head, head_color)

    for l in state.apple_loc:
        draw_object(surface, l, apple_color)

    for o in state.obstacle_loc:
        draw_object(surface, o, obstacle_color)

    for row in range(int(game_width)):
        for col in range(int(game_length)):
            if row == 0 or row == game_width - 1 or col == 0 or col == game_length - 1:
                draw_object(surface, (col, row), obstacle_color)


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


def drawGrid(surface):
    for y in range(0, int(game_width)):
        for x in range(0, int(game_length)):
            if (x + y) % 2 == 0:
                r = pygame.Rect((x * gridsize, y * gridsize), (gridsize, gridsize))
                pygame.draw.rect(surface, (93, 216, 228), r)
            else:
                rr = pygame.Rect((x * gridsize, y * gridsize), (gridsize, gridsize))
                pygame.draw.rect(surface, (84, 194, 205), rr)

def main():
    for i in range(123, 130):
        filepath = rf"C:\Users\ibm\PycharmProjects\Snake\eval_data\Eval_data.step{i}.txt"
        print("Step: ", i)
        print_all(filepath)

if __name__ == "__main__":
    main()
