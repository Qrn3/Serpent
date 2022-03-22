import torch
import random
import time
import numpy as np
from collections import deque
from serpent_game import SerpentGameAI, Direction, Point
from model import Linear_QNetwork, QTrainer
from helper import plot

MAX_MEM = 100_000
BATCH_SIZE = 1000

class Serpent:
	def __init__(self):
		self.n_games = 0
		self.epsilon = 0
		self.gamma = 0.90
		self.lr = 0.001
		self.memory = deque(maxlen=MAX_MEM)
		self.model = Linear_QNetwork(11, 256, 3)
		self.trainer = QTrainer(self.model, self.lr, self.gamma)

	def get_state(self, game):
		head = game.snake[0]
		point_left = Point(head.x - 20, head.y)
		point_right = Point(head.x + 20, head.y)
		point_up = Point(head.x, head.y - 20)
		point_down = Point(head.x, head.y + 20)

		is_dir_l = game.direction == Direction.LEFT
		is_dir_r = game.direction == Direction.RIGHT
		is_dir_u = game.direction == Direction.UP
		is_dir_d = game.direction == Direction.DOWN

		state = [
            (is_dir_r and game.is_collision(point_right)) or 
            (is_dir_l and game.is_collision(point_left)) or 
            (is_dir_u and game.is_collision(point_up)) or 
            (is_dir_d and game.is_collision(point_down)),

            (is_dir_u and game.is_collision(point_right)) or 
            (is_dir_d and game.is_collision(point_left)) or 
            (is_dir_l and game.is_collision(point_up)) or 
            (is_dir_r and game.is_collision(point_down)),

            (is_dir_d and game.is_collision(point_right)) or 
            (is_dir_u and game.is_collision(point_left)) or 
            (is_dir_r and game.is_collision(point_up)) or 
            (is_dir_l and game.is_collision(point_down)),
            
            is_dir_l,
            is_dir_r,
            is_dir_u,
            is_dir_d,
            
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y 
            ]

		return np.array(state, dtype=int)

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def train_long_mem(self):
		if len(self.memory) > BATCH_SIZE: mini_sample = random.sample(self.memory, BATCH_SIZE)
		else: mini_sample = self.memory

		states, actions, rewards, next_states, dones = zip(*mini_sample)
		self.trainer.train_step(states, actions, rewards, next_states, dones)

	def train_short_mem(self, state, action, reward, next_state, done):
		self.trainer.train_step(state, action, reward, next_state, done)

	def get_action(self, state):
		self.epsilon = 80 - self.n_games
		final_move = [0,0,0]
		if random.randint(0, 200) < self.epsilon:
			move = random.randint(0, 2)
			final_move[move] = 1
		else:
			state0 = torch.tensor(state, dtype=torch.float)
			prediction = self.model(state0)
			move = torch.argmax(prediction).item()
			final_move[move] = 1

		return final_move

def train():
	plot_scores = []
	plot_mean_scores = []
	total_score = 0
	record = 0
	agent = Serpent()
	game = SerpentGameAI()
	start = time.time()

	while True:
		old_state = agent.get_state(game)

		final_move = agent.get_action(old_state)

		reward, done, score = game.play_step(final_move)
		new_state = agent.get_state(game)

		agent.train_short_mem(old_state, final_move, reward, new_state, done)

		agent.remember(old_state, final_move, reward, new_state, done)

		if done:
			game.reset()
			agent.n_games += 1
			agent.train_long_mem()

			if score > record:
				record = score
				agent.model.save()

			now = time.time() - start
			time_string = time.ctime(now)[14:19]
			
			print(f"Game: {agent.n_games} \nScore: {score} \nRecord: {record}")
			print(f"Has been: {time_string}\n***************")

			plot_scores.append(score)
			total_score += score
			mean_score = total_score / agent.n_games
			plot_mean_scores.append(mean_score)
			plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
	train()