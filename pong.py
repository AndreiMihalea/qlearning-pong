import pygame
import sys
from random import choice, random
from argparse import ArgumentParser
import time
from copy import copy, deepcopy

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

GRAY_C = (36, 36, 36)
ORANGE_C = (255, 140, 0)

global rows, cols, size

WIDTH = 50
HEIGHT = 50
MARGIN = 5

GAME_WIDTH = 40
GAME_HEIGHT = 40
GAME_MARGIN = 1

ACTIONS = ["UP", "DOWN", "STAY"]
ACTION_EFFECTS = {
    "UP": (-1,0),
    "DOWN": (1,0),
    "STAY": (0,0)
}

MOVE_REWARD = -0.1
WIN_REWARD = 10.0
LOSE_REWARD = -10.0

def __serialize_state(state):
    return "\n".join(map(lambda row: "".join(row), state))

def __deserialize_state(str_state):
    return list(map(list, str_state.split("\n")))

def display_state(state, score):
    width = get_dimensions(state)[1]
    print (width * 'x')
    print (state[-3])
    print (width * 'x')
    if not is_final_state(state, score):
    	print ("Not yet finished")
    	print ("Bounces: " + str(score))
    else:
    	if get_winner(state) == 1:
    		print ("Player 1 wins")
    	elif get_winner(state) == 2:
    		print ("Player 2 wins")
    	else:
    		print ("Draw")

    print (2 * width * '-')

def get_initial_state(rows, cols, size):
	grid = []
	for row in range(rows):
	    grid.append([])
	    for column in range(cols):
	    	grid[row].append(' ')

	paddle1 = [0, 0]
	paddle2 = [0, 0]

	paddle1[0] = rows//2 - size//2
	paddle1[1] = paddle1[0] + size

	paddle2[0] = rows//2 - size//2
	paddle2[1] = paddle1[0] + size

	for row in range(paddle1[0], paddle1[1]):
		grid[row][0] = '2'
		grid[row][cols - 1] = '2'

	if cols % 2 == 0:
		ballx = choice([cols//2 - 1, cols//2])
	else:
		ballx = cols//2

	bally = choice([y for y in range(0, rows)])

	ballpos = [bally, ballx]
	direction = choice([[0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]])

	if ballpos[0] == 0 and direction[0] == -1:
		direction[0] = 1

	if ballpos[0] == rows - 1 and direction[0] == 1:
		direction[0] = -1

	sp1 = (paddle1[0], ballpos[0], ballpos[1])
	sp2 = (paddle2[0], ballpos[0], ballpos[1])

	return [grid, ballpos, direction, paddle1, paddle2, __serialize_state(grid), sp1, sp2]

def get_legal_actions(state, player):
	grid = state[0]
	rows = len(grid)
	cols = len(grid[0])
	legal_actions = []
	if player == 1:
		c = 0
	else:
		c = cols - 1

	column = [x[c] for x in grid]
	if column[0] != '2' and column[rows - 1] != '2':
		legal_actions = ["UP", "DOWN", "STAY"]
	if column[0] != '2' and column[rows - 1] == '2':
		legal_actions = ["UP", "STAY"]
	if column[0] == '2' and column[rows - 1] != '2':
		legal_actions = ["DOWN", "STAY"]
	if column[0] == '2' and column[rows - 1] == '2':
		legal_actions = ["STAY"]

	return legal_actions

def is_final_state(state, score):
	rows, cols, size = get_dimensions(state)
	ballpos = state[1]

	if score > 20:
		return True

	if ballpos[1] == 0 or ballpos[1] == cols - 1:
		return True

	return False

def get_winner(state):
	rows, cols, size = get_dimensions(state)
	ballpos = state[1]

	if ballpos[1] == 0:
		return 2

	if ballpos[1] == cols - 1:
		return 1

def get_dimensions(state):
	grid = state[0]
	paddle1 = state[3]
	rows = len(grid)
	cols = len(grid[0])
	size = paddle1[1] - paddle1[0]

	return rows, cols, size

def perfect(state, legal_actions, alpha):
    chances = int(1000 * alpha) * [0] + int(1000 * (1 - alpha)) * [1]
    r = choice(chances)

    _, _, size = get_dimensions(state)

    if r == 0:
    	return choice(legal_actions)
    else:
    	ballpos = state[1]
    	paddle2 = state[4]

    	perf = move_ball(state)

    	action = 'STAY'

    	if perf is not None and perf < paddle2[0] + size // 2 and 'UP' in legal_actions:
    		action = 'UP'

    	elif perf is not None and perf > paddle2[1] - size // 2 - 1 and 'DOWN' in legal_actions:
    		action = 'DOWN'

    	return action

def apply_action(state, action, adv_action, current_score):
	reward = 0
	rows, cols, size = get_dimensions(state)

	grid = state[0]
	ballpos = state[1]
	direction = state[2]
	paddle1 = state[3]
	paddle2 = state[4]

	for row in range(rows):
		for column in range(cols):
			if grid[row][column] == '2':
				grid[row][column] = ' '

	if action == 'UP':
		paddle1[0] -= 1
		paddle1[1] -= 1
		reward -= 0.1
	if action == 'DOWN':
		paddle1[0] += 1
		paddle1[1] += 1
		reward -= 0.1

	if adv_action == 'UP':
		paddle2[0] -= 1
		paddle2[1] -= 1
	if adv_action == 'DOWN':
		paddle2[0] += 1
		paddle2[1] += 1

	for row in range(paddle1[0], paddle1[1]):
		grid[row][0] = '2'

	for row in range(paddle2[0], paddle2[1]):
		grid[row][cols - 1] = '2'

	if ballpos[0] >= 0 and ballpos[0] <= rows - 1 and ballpos[1] >= 0 and ballpos[1] <= cols - 1:
	    ballpos[0] += direction[0]
	    ballpos[1] += direction[1]

	if ballpos[0] == 0 or ballpos[0] == rows - 1:
		direction[0] *= -1

	if ballpos[1] == 1 and ballpos[0] >= paddle1[0] and ballpos[0] < paddle1[1]:
		direction[1] *= -1
		reward += 1

	if ballpos[1] == cols - 2 and ballpos[0] >= paddle2[0] and ballpos[0] < paddle2[1]:
		direction[1] *= -1
		current_score += 1

	if ballpos[1] == 0 or ballpos[1] == cols - 1:
		if ballpos[1] == 0:
			reward -= 10.0

	for row in range(rows):
		for column in range(cols):
			if grid[row][column] == '*':
				grid[row][column] = ' '

	for row in range(rows):
		for column in range(cols):
			if row == ballpos[0] and column == ballpos[1]:
				grid[row][column] = '*'

	sp1 = (paddle1[0], ballpos[0], ballpos[1])
	sp2 = (paddle2[0], ballpos[0], ballpos[1])

	return [grid, state[1], state[2], paddle1, paddle2, __serialize_state(grid), sp1, sp2], reward, current_score

def move_ball(state):
	rows, cols, size = get_dimensions(state)
	grid = copy(state[0])
	ballpos = copy(state[1])
	direction = copy(state[2])

	while ballpos[1] != cols - 2:
		if ballpos[0] >= 0 and ballpos[0] <= rows - 1 and ballpos[1] >= 0 and ballpos[1] <= cols - 1:
			ballpos[0] += direction[0]
			ballpos[1] += direction[1]

		if ballpos[0] == 0 or ballpos[0] == rows - 1:
			direction[0] *= -1

		if ballpos[1] == 1:
			direction[1] *= -1

		if ballpos[1] == 0:
			return None

		if ballpos[1] == cols - 2:
			return ballpos[0]

		ballpos += direction


def keydown(event, state):
    grid = state[0]
    paddle1 = state[3]
    paddle2 = state[4]

    actions1 = get_legal_actions(state, 1)
    actions2 = get_legal_actions(state, 2)

    new_state = None

    if event.key == pygame.K_UP and 'UP' in actions2:
    	new_state = apply_action(state, 'UP', 2)
    elif event.key == pygame.K_DOWN and 'DOWN' in actions2:
    	new_state = apply_action(state, 'DOWN', 2)
    elif event.key == pygame.K_w and 'UP' in actions1:
    	new_state = apply_action(state, 'UP', 1)
    elif event.key == pygame.K_s and 'DOWN' in actions1:
    	new_state = apply_action(state, 'DOWN', 1)

    if new_state:
    	return new_state
    else:
    	return state


def draw(state, train_ep, args, bounces):
    rows, cols, size = get_dimensions(state)

    window_size = [41 * cols + 1, 41 * (rows + 7) + 1]
    screen = pygame.display.set_mode(window_size)
	 
    pygame.display.set_caption("Array Backed Grid")

    done = False

    clock = pygame.time.Clock()

    grid = state[0]
    ballpos = state[1]
    direction = state[2]
    paddle1 = state[3]
    paddle2 = state[4]

    screen.fill(GRAY_C)

    for row in range(rows):
        for column in range(cols):
            if row == ballpos[0] and column == ballpos[1]:
            	pygame.draw.rect(screen,
	                             WHITE,
	                             [(GAME_MARGIN + GAME_WIDTH) * column + GAME_MARGIN,
	                              (GAME_MARGIN + GAME_HEIGHT) * row + GAME_MARGIN,
	                              GAME_WIDTH,
	                              GAME_HEIGHT])
            	pygame.draw.circle(screen,
                				 RED,
                				 [(GAME_MARGIN + GAME_WIDTH) * column + GAME_MARGIN + 20,
	                              (GAME_MARGIN + GAME_HEIGHT) * row + GAME_MARGIN + 20],
	                              20)
            elif grid[row][column] == '2':
            	pygame.draw.rect(screen,
	                             BLACK,
	                             [(GAME_MARGIN + GAME_WIDTH) * column + GAME_MARGIN,
	                              (GAME_MARGIN + GAME_HEIGHT) * row + GAME_MARGIN,
	                              GAME_WIDTH,
	                              GAME_HEIGHT])
            else:
	            pygame.draw.rect(screen,
	                             WHITE,
	                             [(GAME_MARGIN + GAME_WIDTH) * column + GAME_MARGIN,
	                              (GAME_MARGIN + GAME_HEIGHT) * row + GAME_MARGIN,
	                              GAME_WIDTH,
	                              GAME_HEIGHT])

    smallText = pygame.font.Font("freesansbold.ttf", 18)

    textSurf, textRect = text_objects("Learning Rate: " + str(args.learning_rate), smallText, WHITE)
    textRect.center = ((cols * 40 + 40 * GAME_MARGIN) / 2, 50 * row)
    screen.blit(textSurf, textRect)

    textSurf, textRect = text_objects("Discount: " + str(args.discount), smallText, WHITE)
    textRect.center = ((cols * 40 + 40 * GAME_MARGIN) / 2, 50 * row + 30)
    screen.blit(textSurf, textRect)

    textSurf, textRect = text_objects("Training Episodes: " + str(args.train_episodes), smallText, WHITE)
    textRect.center = ((cols * 40 + 40 * GAME_MARGIN) / 2, 50 * row + 60)
    screen.blit(textSurf, textRect)

    textSurf, textRect = text_objects("Agent Type: " + str(args.agent), smallText, WHITE)
    textRect.center = ((cols * 40 + 40 * GAME_MARGIN) / 2, 50 * row + 90)
    screen.blit(textSurf, textRect)

    textSurf, textRect = text_objects("Epsilon: " + str(args.epsilon), smallText, WHITE)
    textRect.center = ((cols * 40 + 40 * GAME_MARGIN) / 2, 50 * row + 120)
    screen.blit(textSurf, textRect)

    textSurf, textRect = text_objects("Enemy Type: " + str(args.enemy), smallText, WHITE)
    textRect.center = ((cols * 40 + 40 * GAME_MARGIN) / 2, 50 * row + 150)
    screen.blit(textSurf, textRect)

    textSurf, textRect = text_objects("Alpha: " + str(args.alpha), smallText, WHITE)
    textRect.center = ((cols * 40 + 40 * GAME_MARGIN) / 2, 50 * row + 180)
    screen.blit(textSurf, textRect)

    textSurf, textRect = text_objects("Bounces: " + str(bounces), smallText, WHITE)
    textRect.center = ((cols * 40 + 40 * GAME_MARGIN) / 2, 50 * row + 210)
    screen.blit(textSurf, textRect)

    if train_ep is not None:
	    textSurf, textRect = text_objects("Training Episode: " + str(train_ep) + "/" + str(args.train_episodes), smallText, WHITE)
	    textRect.center = ((cols * 40 + 40 * GAME_MARGIN) / 2, 50 * row + 240)
	    screen.blit(textSurf, textRect)

    clock.tick(60)
 
    pygame.display.flip()


def text_objects(text, font, color):
    textSurface = font.render(text, True, color)
    return textSurface, textSurface.get_rect()


def draw_gui(args):
	grid = []
	rows = 7
	cols = 8
	for row in range(rows):
		grid.append([])
		for column in range(cols):
			grid[row].append(0)

	pygame.init()

	window_size = [55 * cols + MARGIN, 55 * (rows + 1) + MARGIN]
	screen = pygame.display.set_mode(window_size)

	pygame.display.set_caption("Array Backed Grid")

	font = pygame.font.SysFont('Arial', 25)

	done = False
	clock = pygame.time.Clock()

	parameters = ['Learning Rate', 'Discount', 'Training Episodes', 'Agent Type', 'Epsilon', 'Enemy Type', 'Alpha']
	values = [args.learning_rate, args.discount, args.train_episodes, args.agent, args.epsilon, args.enemy, args.alpha]
	nexta = {'random' : 'greedy', 'greedy' : 'egreedy', 'egreedy' : 'random'}
	preva = {'random' : 'egreedy', 'greedy' : 'random', 'egreedy' : 'greedy'}

	nexte = {'random' : 'greedy', 'greedy' : 'perfect', 'perfect' : 'random'}
	preve = {'random' : 'perfect', 'greedy' : 'random', 'perfect' : 'greedy'}

	step = {0 : 0.01, 1 : 0.01, 2 : 100, 4 : 0.01, 6 : 0.01}

	limits = {0 : [0, 1], 1 : [0, 1], 2 : [0, 100000], 4 : [0, 1], 6 : [0, 1]}

	start = False
	 
	while not done:
	    for event in pygame.event.get():
	        if event.type == pygame.QUIT:
	            done = True
	        elif event.type == pygame.MOUSEBUTTONDOWN:
	            pos = pygame.mouse.get_pos()
	            column = pos[0] // (WIDTH + MARGIN)
	            row = pos[1] // (HEIGHT + MARGIN)
	            if row < rows and (column == 3 or column == 7):
	            	grid[row][column] = 1
	            if row == rows:
	            	start = True
	 
	    screen.fill(BLACK)
	 
	    for row in range(rows):
	        for column in [3, 7]:
	            color = GRAY_C
	            if grid[row][column] == 1:
	            	color = ORANGE_C
	            	if column == 3:
	            		if row == 3:
	            			values[row] = preva[values[row]]
	            		elif row == 5:
	            			values[row] = preve[values[row]]
	            		else:
	            			if values[row] - step[row] >= limits[row][0]:
	            				values[row] -= step[row]
	            				if row != 2:
	            					values[row] = float("{0:.2f}".format(values[row]))

	            	if column == 7:
	            		if row == 3:
	            			values[row] = nexta[values[row]]
	            		elif row == 5:
	            			values[row] = nexte[values[row]]
	            		else:
	            			if values[row] + step[row] <= limits[row][1]:
	            				values[row] += step[row]
	            				if row != 2:
	            					values[row] = float("{0:.2f}".format(values[row]))

	            	grid[row][column] = 0

	            pygame.draw.rect(screen,
	                             color,
	                             [(MARGIN + WIDTH) * column + MARGIN,
	                              (MARGIN + HEIGHT) * row + MARGIN,
	                              WIDTH,
	                              HEIGHT])

	        smallText = pygame.font.Font("freesansbold.ttf", 16)

	        textSurf, textRect = text_objects("<", smallText, WHITE)
	        textRect.center = (170 + 50 / 2, 5 * (row + 1) + 50 * row + 50 / 2)
	        screen.blit(textSurf, textRect)

	        textSurf, textRect = text_objects(">", smallText, WHITE)
	        textRect.center = (390 + 50 / 2, 5 * (row + 1) + 50 * row + 50 / 2)
	        screen.blit(textSurf, textRect)

	    for row in range(rows):
	    	for column in range(0, 1):
	    		pygame.draw.rect(screen,
	                             GRAY_C,
	                             [(MARGIN + WIDTH) * column + MARGIN,
	                              (MARGIN + HEIGHT) * row + MARGIN,
	                              3 * WIDTH + 2 * MARGIN,
	                              HEIGHT])
	    		
	    		textSurf, textRect = text_objects(parameters[row], smallText, WHITE)
	    		textRect.center = (170 / 2, 5 * (row + 1) + 50 * row + 50 / 2)
	    		screen.blit(textSurf, textRect)

	    for row in range(rows):
	    	for column in range(4, 5):
	    		pygame.draw.rect(screen,
	                             GRAY_C,
	                             [(MARGIN + WIDTH) * column + MARGIN,
	                              (MARGIN + HEIGHT) * row + MARGIN,
	                              3 * WIDTH + 2 * MARGIN,
	                              HEIGHT])
	    		
	    		textSurf, textRect = text_objects(str(values[row]), smallText, WHITE)
	    		textRect.center = (225 + 150 / 2, 5 * (row + 1) + 50 * row + 50 / 2)
	    		screen.blit(textSurf, textRect)

	    if start == False:
	    	color_start = ORANGE_C
	    else:
	    	color_start = WHITE

	    pygame.draw.rect(screen,
                         color_start,
                         [(MARGIN + WIDTH) * 0 + MARGIN,
                          (MARGIN + HEIGHT) * rows + MARGIN,
                          8 * WIDTH + 7 * MARGIN,
                          HEIGHT])

	    textSurf, textRect = text_objects("START!", smallText, BLACK)
	    textRect.center = ((9 * MARGIN + 8 * WIDTH) / 2, 5 * (rows + 1) + 50 * (rows) + 50 / 2)
	    screen.blit(textSurf, textRect)
	 
	    clock.tick(60)

	    time.sleep(0.1)

	    pygame.display.flip()

	    if start == True:
	    	done = True

	pygame.quit()

	return values
