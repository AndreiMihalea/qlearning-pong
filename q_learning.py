# General imports
from copy import copy
from random import choice, random
from argparse import ArgumentParser
from time import sleep

# Game functions
from pong import *

def epsilon_greedy(Q, state, legal_actions, player, epsilon):
    chances = int(1000 * epsilon) * [0] + int(1000 * (1 - epsilon)) * [1]
    r = choice(chances)
    if r == 0:
        return choice(legal_actions)
    else:
        maximum = -float('inf')
        argmax = legal_actions[0]

        if player == 1:
        	grid = state[-2]
        else:
        	_, cols, _ = get_dimensions(state)
        	grid = state[-1]
        	grid = list(grid)
        	grid[2] = cols - 1 - grid[2]
        	grid = tuple(grid)

        for action in legal_actions:
            if Q.get((grid, action), 0) > maximum:
                maximum = Q.get((grid, action), 0)
                argmax = action

        return argmax

def best_action(Q, state, legal_actions, player):
    maximum = -float('inf')
    argmax = legal_actions[0]

    if player == 1:
        	grid = state[-2]
    else:
    	_, cols, _ = get_dimensions(state)
    	grid = state[-1]
    	grid = list(grid)
    	grid[2] = cols - 1 - grid[2]
    	grid = tuple(grid)

    for action in legal_actions:
        if Q.get((grid, action), 0) > maximum:
            maximum = Q.get((grid, action), 0)
            argmax = action

    return argmax

def q_learning(args):
    Q = {}
    train_scores = []
    eval_scores = []
                                                          # for each episode ...
    for train_ep in range(1, args.train_episodes + 1):

                                                    # ... get the initial state,
        score = 0
        state = get_initial_state(args.grid_height, args.grid_width, args.paddle_size)
        current_score = 0
        
        if args.draw:
        	pygame.init()

                                           # while current state is not terminal
        while True:

            if args.verbose:
        	    display_state(state, current_score); sleep(args.sleep)

            if args.draw:
        	    draw(state, train_ep, args, current_score)
        	    if not args.verbose:
        	        sleep(args.sleep)

                                               # choose one of the legal actions
            actions = get_legal_actions(state, 1)
            if args.agent == 'random':
            	action = epsilon_greedy(Q, state, actions, 1, 1)
            elif args.agent == 'greedy':
            	action = epsilon_greedy(Q, state, actions, 1, 0)
            elif args.agent == 'egreedy':
            	action = epsilon_greedy(Q, state, actions, 1, args.epsilon)

                            # apply action and get the next state and the reward
            if args.enemy == 'random':
            	adv_action = choice(get_legal_actions(state, 2))
            elif args.enemy == 'greedy':
            	actions2 = get_legal_actions(state, 2)
            	adv_action = best_action(Q, state, actions2, 2)
            elif args.enemy == 'perfect':
            	actions2 = get_legal_actions(state, 2)
            	adv_action = perfect(state, actions2, args.alpha)

            statep, reward, current_score = apply_action(state, action, adv_action, current_score)
            gridp = statep[-2]
            score += reward

            # Q-Learning
            maximum = -float('inf')
            actionsp = get_legal_actions(statep, 1)
            for actionp in actionsp:
                if Q.get((gridp, actionp), 0) > maximum:
                    maximum = Q.get((gridp, actionp), 0)

            grid = state[-2]
            Q[(grid, action)] = Q.get((grid, action), 0) + args.learning_rate * (reward + args.discount * maximum - Q.get((grid, action), 0))

            state = statep

            if is_final_state(state, current_score):
            	if args.verbose:
            		display_state(state, current_score); sleep(args.sleep)
            	break

        print("Episode %6d / %6d" % (train_ep, args.train_episodes) + " " + str(score))
        train_scores.append(score)

                                                    # evaluate the greedy policy
        if train_ep % args.eval_every == 0:
            avg_score = .0

            # Evaluate
            
            avg_score = 0
            for i in range(10):
                score = 0
                current_score = 0
                state = get_initial_state(args.grid_height, args.grid_width, args.paddle_size)
                while not is_final_state(state, current_score):
                    action = best_action(Q, state, get_legal_actions(state, 1), 1)

                    if args.enemy == 'random':
                    	adv_action = choice(get_legal_actions(state, 2))
                    elif args.enemy == 'greedy':
                    	actions2 = get_legal_actions(state, 2)
                    	adv_action = best_action(Q, state, actions2, 2)
                    elif args.enemy == 'perfect':
                    	actions2 = get_legal_actions(state, 2)
                    	adv_action = perfect(state, actions2, args.alpha)

                    state, reward, current_score = apply_action(state, action, adv_action, current_score)
                    score += reward

                avg_score += score

            avg_score /= 10

            eval_scores.append(avg_score)



    # --------------------------------------------------------------------------
    nr1 = 0
    nr2 = 0
    nr3 = 0
    for i in range(10):
        state = get_initial_state(args.grid_height, args.grid_width, args.paddle_size)
        final_score = 0
        current_score = 0
        while not is_final_state(state, current_score):
            action = best_action(Q, state, get_legal_actions(state, 1), 1)

            if args.enemy == 'random':
            	adv_action = choice(get_legal_actions(state, 2))
            elif args.enemy == 'greedy':
            	actions2 = get_legal_actions(state, 2)
            	adv_action = best_action(Q, state, actions2, 2)
            elif args.enemy == 'perfect':
            	actions2 = get_legal_actions(state, 2)
            	adv_action = perfect(state, actions2, args.alpha)

            state, reward, current_score = apply_action(state, action, adv_action, current_score)
            final_score += reward

        if get_winner(state) == 1:
        	nr1 += 1
        elif get_winner(state) == 2:
        	nr2 += 1
        else:
        	nr3 += 1

    print (nr1)
    print (nr2)
    print (nr3)

    if args.final_show:
        state = get_initial_state(args.grid_height, args.grid_width, args.paddle_size)
        final_score = 0
        current_score = 0
        if args.final_draw and not args.draw:
        	pygame.init()
        while True:
            action = best_action(Q, state, get_legal_actions(state, 1), 1)

            if args.final_draw:
            	draw(state, None, args, current_score)

            display_state(state, current_score); sleep(args.sleep)

            if args.enemy == 'random':
            	adv_action = choice(get_legal_actions(state, 2))
            elif args.enemy == 'greedy':
            	actions2 = get_legal_actions(state, 2)
            	adv_action = best_action(Q, state, actions2, 2)
            elif args.enemy == 'perfect':
            	actions2 = get_legal_actions(state, 2)
            	adv_action = perfect(state, actions2, args.alpha)

            state, reward, current_score = apply_action(state, action, adv_action, current_score)
            final_score += reward

            if is_final_state(state, current_score):
            	draw(state, None, args, current_score)
            	display_state(state, current_score); sleep(args.sleep)
            	break

        if args.final_draw:
         	pygame.quit()

    if args.final_draw and not args.final_show:
        if not args.draw:
        	pygame.init()
        state = get_initial_state(args.grid_height, args.grid_width, args.paddle_size)
        final_score = 0
        current_score = 0
        while True:
            action = best_action(Q, state, get_legal_actions(state, 1), 1)

            draw(state, None, args, current_score); sleep(args.sleep)

            if args.enemy == 'random':
            	adv_action = choice(get_legal_actions(state, 2))
            elif args.enemy == 'greedy':
            	actions2 = get_legal_actions(state, 2)
            	adv_action = best_action(Q, state, actions2, 2)
            elif args.enemy == 'perfect':
            	actions2 = get_legal_actions(state, 2)
            	adv_action = perfect(state, actions2, args.alpha)

            state, reward, current_score = apply_action(state, action, adv_action, current_score)
            final_score += reward

            if is_final_state(state, current_score):
            	break
        pygame.quit()

    if args.draw and not args.final_draw:
    	pygame.quit()

    return train_scores, eval_scores, args.enemy, nr1, nr2, nr3

if __name__ == "__main__":
    parser = ArgumentParser()
    # Input file
    parser.add_argument("--agent", type = str.lower, default = "random",
                        help = "Agent type")
    parser.add_argument("--enemy", type = str.lower, default = "random",
                        help = "Enemy type")
    parser.add_argument("--grid_height", type = int, default = 9,
                        help = "Grid height")
    parser.add_argument("--grid_width", type = int, default = 16,
                        help = "Grid width")
    parser.add_argument("--paddle_size", type = int, default = 3,
                        help = "Paddle size")
    parser.add_argument("--alpha", type = float, default = 0.03,
                        help = "Alpha perfect")
    # Meta-parameters
    parser.add_argument("--learning_rate", type = float, default = 0.1,
                        help = "Learning rate")
    parser.add_argument("--discount", type = float, default = 0.99,
                        help = "Value for the discount factor")
    parser.add_argument("--epsilon", type = float, default = 0.05,
                        help = "Probability to choose a random action.")
    # Training and evaluation episodes
    parser.add_argument("--train_episodes", type = int, default = 1000,
                        help = "Number of episodes")
    parser.add_argument("--eval_every", type = int, default = 10,
                        help = "Evaluate policy every ... games.")
    parser.add_argument("--eval_episodes", type = float, default = 10,
                        help = "Number of games to play for evaluation.")
    # Display
    parser.add_argument("--verbose", dest="verbose",
                        action = "store_true", help = "Print each state")
    parser.add_argument("--draw", dest="draw",
                        action = "store_true", help = "Draw each state")
    parser.add_argument("--final_draw", dest="final_draw",
                        action = "store_true", help = "Draw final state")
    parser.add_argument("--gui", dest="gui",
                        action = "store_true", help = "GUI")
    parser.add_argument("--plot", dest="plot_scores", action="store_true",
                        help = "Plot scores in the end")
    parser.add_argument("--sleep", type = float, default = 0.1,
                        help = "Seconds to 'sleep' between moves.")
    parser.add_argument("--final_show", dest = "final_show",
                        action = "store_true",
                        help = "Demonstrate final strategy.")
    args = parser.parse_args()

    if args.gui:
    	[args.learning_rate, args.discount, args.train_episodes, args.agent, args.epsilon,
    	args.enemy, args.alpha] = draw_gui(args)
    	args.agent = args.agent.lower()
    	args.enemy = args.enemy.lower()

    print (args)

    q_learning(args)
    '''
    all_data = []
    for i in [(9, 16, 3), (14, 30, 3), (14, 30, 6), (20, 20, 3)]:
    	args.grid_height = i[0]
    	args.grid_width = i[1]
    	args.paddle_size = i[2]
    	all_data.append(q_learning(args))

    if args.plot_scores:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        import matplotlib.ticker as ticker
        import numpy as np
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Training Episodes')
        ax.set_zlabel('Scores')
        plt.xticks([0,1,2,3], ['(9, 16, 3)', '(14, 30, 3)', '(14, 30, 6)', '(20, 20, 3)'])
        for i, d in enumerate(all_data):
        	print (str(d[3]) + " " + str(d[4]) + " " + str(d[5]))
        	print ()
        	if i == 0:
        		c = "blue"
        	elif i == 1:
        		c = "green"
        	elif i == 2:
        		c = "orange"
        	elif i == 3:
        		c = "black"
        	train_scores = d[0]
        	eval_scores = d[1]
	        a = np.empty(args.train_episodes)
	        a.fill(i)
	        plt.plot(
	        	a,
	            np.linspace(1, args.train_episodes, args.train_episodes),
	            np.convolve(train_scores, [0.2,0.2,0.2,0.2,0.2], "same"),
	            linewidth = 1.0, color = c
	        )
	        a = np.empty(len(eval_scores))
	        a.fill(i)
	        plt.plot(
	        	a,
	            np.linspace(args.eval_every, args.train_episodes, len(eval_scores)),
	            eval_scores, linewidth = 2.0, color = "red"
	        )
        plt.show()
    '''
    '''
    train_scores, eval_scores, _, _, _, _ = q_learning(args)
    from matplotlib import pyplot as plt
    import numpy as np
    plt.xlabel("Training episodes")
    plt.ylabel("Average score")
    plt.plot(
        np.linspace(1, args.train_episodes, args.train_episodes),
        np.convolve(train_scores, [0.2,0.2,0.2,0.2,0.2], "same"),
        linewidth = 1.0, color = "blue"
    )
    plt.plot(
        np.linspace(args.eval_every, args.train_episodes, len(eval_scores)),
        eval_scores, linewidth = 2.0, color = "red"
    )
    plt.show()
    '''