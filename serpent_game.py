import pygame
import random 
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()
myfont = pygame.font.SysFont("calibri",40)

Point = namedtuple("Point", "x, y")
SPEED = 40

#RGB colours
WHITE 	= (255, 255, 255)
BLACK 	= (0, 0, 0)
GREEN1	= (0, 255, 0)
GREEN2	= (0, 100, 255)
RED 	= (255, 0, 0)

class Direction(Enum):
	RIGHT 	= 1
	LEFT 	= 2
	UP 		= 3
	DOWN 	= 4

class SerpentGameAI:

	def __init__(self, w=640, h=480):
		self.w = w
		self.h = h
		self.display = pygame.display.set_mode((self.w, self.h))
		pygame.display.set_caption("Serpent")
		self.clock = pygame.time.Clock()
		self.reset()

	def reset(self):
		self.direction = Direction.RIGHT
		self.head = Point(self.w/2, self.h/2)
		self.snake = [	self.head,
						Point(self.head.x-20, self.head.y),
						Point(self.head.x-(2*20), self.head.y)]

		self.score = 0
		self.food = None
		self.place_food()
		self.frame_iteration = 0

	def place_food(self):
		x = random.randint(0, (self.w-20)//20)*20 
		y = random.randint(0, (self.h-20)//20)*20 

		self.food = Point(x, y)
		if self.food in self.snake: self.place_food()

	def play_step(self, action):
		self.frame_iteration += 1
		#Collect User Inputs
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()

		#Move
		self._move(action)
		self.snake.insert(0, self.head)

		#Is Game Over?
		reward = 0
		game_over = False
		if self.is_collision() or self.frame_iteration > 100*len(self.snake):
			game_over = True
			reward = -10
			return reward, game_over, self.score

		#Place New Food and/or Move
		if self.head == self.food:
			self.score += 1
			reward = 10
			self.place_food()
		else: self.snake.pop()

		#UI Update
		self._update_ui()
		self.clock.tick(SPEED)

		#Return Data
		return reward, game_over, self.score

	def is_collision(self, pt=None):
		if pt is None: pt = self.head
		if pt.x > self.w-20 or pt.x < 0 or pt.y > self.h-20 or pt.y < 0: return True
		if pt in self.snake[1:]: return True
		return False

	def _update_ui(self):
		self.display.fill(BLACK)

		for pt in self.snake:
			pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, 20, 20))
			pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

		pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, 12, 12))

		text = myfont.render(f"Score: {str(self.score)}", True, WHITE)
		self.display.blit(text, [0,0])
		pygame.display.flip()

	def _move(self, action):
		# [s,r,l]
		clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
		idx = clock_wise.index(self.direction)

		if np.array_equal(action, [1, 0, 0]): new_dir = clock_wise[idx]
		elif np.array_equal(action, [0, 1, 1]):	new_dir = clock_wise[(idx+1) % 4]
		else: new_dir = clock_wise[(idx-1) % 4]
		
		self.direction = new_dir

		x = self.head.x		
		y = self.head.y

		if self.direction == Direction.RIGHT: x += 20
		elif self.direction == Direction.LEFT: x -= 20
		elif self.direction == Direction.DOWN: y += 20
		elif self.direction == Direction.UP: y -= 20

		self.head = Point(x, y)