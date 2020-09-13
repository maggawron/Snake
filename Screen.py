import pygame


class Screen:
    snake_color = (17, 24, 47)
    head_color = (255, 1, 1)
    apple_color = (223, 163, 49)
    obstacle_color = (150, 150, 150)
    grid1_color = (93, 216, 228)
    grid2_color = (84, 194, 205)
    
    def __init__(self, width, length):
        self.game_width = width
        self.game_length = length
        self.gridsize = 20
    
        self.game_display = pygame.display.set_mode((self.game_width * self.gridsize, self.game_length * self.gridsize), 0, 32)
        self.surface = pygame.Surface(self.game_display.get_size()).convert()
        self._drawGrid()
        self.myfont = pygame.font.SysFont("monospace", 15)

    def display_move(self, stan):
        self._drawGrid()
        self._draw(stan)
        self.game_display.blit(self.surface, (0, 0))

    def _draw_object(self, obj, color):
        r = pygame.Rect((obj[1] * self.gridsize, obj[0] * self.gridsize), (self.gridsize, self.gridsize))
        pygame.draw.rect(self.surface, color, r)
        pygame.draw.rect(self.surface, Screen.grid1_color, r, 1)
    
    
    def _draw(self, state):
        for p in state.snake_loc[:-1]:
            self._draw_object(p, Screen.snake_color)
        head = state.snake_loc[-1]
        self._draw_object(head, Screen.head_color)
    
        for l in state.apple_loc:
            self._draw_object(l, Screen.apple_color)
    
        for o in state.obstacle_loc:
            self._draw_object(Screen.obstacle_color)
    
        for row in range(int(self.game_width)):
            for col in range(int(self.game_length)):
                if row == 0 or row == self.game_width - 1 or col == 0 or col == self.game_length - 1:
                    self._draw_object((col, row), Screen.obstacle_color)
    
    
    def _drawGrid(self):
        for y in range(0, int(self.game_width)):
            for x in range(0, int(self.game_length)):
                if (x + y) % 2 == 0:
                    r = pygame.Rect((x * self.gridsize, y * self.gridsize), (self.gridsize, self.gridsize))
                    pygame.draw.rect(self.surface, Screen.grid1_color, r)
                else:
                    rr = pygame.Rect((x * self.gridsize, y * self.gridsize), (self.gridsize, self.gridsize))
                    pygame.draw.rect(self.surface, Screen.grid2_color, rr)

    # TODO print curent reward of the snake
    # print(f"Game over | Points: {stan.points} | Level: {stan.level}")
