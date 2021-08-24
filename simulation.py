###########################
# Variables and Imports #
###########################
from random import random, seed, shuffle, uniform
from math import ceil, sqrt
from functools import reduce
from time import time
import pygame
from pygame import gfxdraw


# if you get an error related to tkinter, run `brew install python-tk`
# for graphics we are using graphics.py to install run `pip install graphics.py`
num_particles = 100
num_steps = 1000
x_lim = 1e-8
y_lim = 1e-8
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

###########################
# MATH FUNCTIONS #
###########################
def distance_point_to_wall(WALL_START, WALL_END, x, y):
    wall_start_x, wall_start_y = WALL_START
    wall_end_x, wall_end_y = WALL_END
    return abs((wall_end_x - wall_start_x)*(wall_start_y-y)-(wall_start_x-x)*(wall_end_y-wall_start_y))/sqrt((wall_end_x-wall_start_x)**2 + (wall_end_y-wall_start_y)**2)

def closer_than_radius(distance):
    if abs(distance) <= Particle.radius:
        return 1
    else:
        return 0

###########################
# Graphics #
###########################
BLUE = [0, 0, 255]
WHITE = [255, 255, 255]
def draw_circle(window, x, y, radius, color):
    # Draw anti-aliased circle
    gfxdraw.aacircle(window, x, y, radius, color)
    gfxdraw.filled_circle(window, x, y, radius, color)
            

###########################
# Particle Representation #
###########################

class Particle(object):
    """Representation of a single particle in the simulation. 
    
    Each particle has a 2D position, velocity, and acceleration, and interacts with other nearby
    particles and the walls. For now, only elastic repulsive collisions.
    
    Mass = 1.00784 AMU
    
    Raidus a_0 = 5.29177210903e-11
    
    """
    mass = 1.00784

    scale_pos = None

    radius = 5.29177210903e-11

    dt = 0.05

    def __init__(self, x, y, vx, vy, ax, ay):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay
        self.oldx = None
        self.oldy = None

    def draw(self, window, rad):
            """
            Create a graphical representation of this particle for visualization.
            win is the graphics windown in which the particle should be drawn.
            
            rad is the radius of the particle"""
            x_coord = int(self.x/x_lim * WINDOW_WIDTH)
            y_coord = int(self.y/y_lim * WINDOW_HEIGHT)
            radius = int(self.radius/x_lim * WINDOW_HEIGHT)
            draw_circle(window, x_coord, y_coord, radius, BLUE)

    def move(self):
        self.oldx, self.oldy = self.x, self.y
        self.vx += self.ax * self.dt
        self.vy += self.ay * self.dt
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
    
    def compute_wall_forces(self):
        TOP_LEFT = (0, y_lim)
        TOP_RIGHT = (x_lim, y_lim)
        BOTTOM_LEFT = (0, 0)
        BOTTOM_RIGHT = (0, x_lim)
        horizontal_count = 0
        vertical_count = 0

        #compute distances
        d_left_wall = distance_point_to_wall(BOTTOM_LEFT, TOP_LEFT, self.x, self.y)
        d_right_wall = distance_point_to_wall(BOTTOM_RIGHT, TOP_RIGHT, self.x, self.y)
        d_bottom_wall = distance_point_to_wall(BOTTOM_LEFT, BOTTOM_RIGHT, self.x, self.y)
        d_top_wall = distance_point_to_wall(BOTTOM_LEFT, BOTTOM_RIGHT, self.x, self.y)

        # check if any particles satisfies close condition
        left_wall = closer_than_radius(d_left_wall)
        right_wall = closer_than_radius(d_right_wall)
        top_wall = closer_than_radius(d_top_wall)
        bottom_wall = closer_than_radius(d_bottom_wall)

        ## First check horizontal walls
        vertical_count = d_left_wall + d_right_wall
        horizontal_count = d_bottom_wall + d_top_wall

        # corner
        if vertical_count + horizontal_count > 1:
            # here we need to compute new vx and vy and convert to ax and ay
            




##################
# Initialization #
##################
def make_particles(n):
    """Construct a list of n particles in two dimensions, initially distributed
    evenly but with random velocities. The resulting list is not spatially
    sorted."""
    seed(1000)
    particles = [Particle(0, 0, 0, 0, 0, 0) for _ in range(n)]

    # Make sure particles are not spatially sorted
    shuffle(particles)

    for p in particles:
        # Distribute particles randomly in our box
        x_coord = uniform(0, x_lim)
        y_coord = uniform(0, y_lim)
        p.x = uniform(0, x_lim)
        p.y = uniform(0, y_lim)

        # Assign random velocities within a bound
        p.vx = random() * 2 - 1
        p.vy = random() * 2 - 1

    return particles

def init_graphics(particles):
    background_colour = WHITE
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('Particle Simulation')
    window.fill(background_colour)
    return window

    



#####################
# Serial Simulation #
#####################
def serial_simulation(n, steps, update_interval=1):

    # Create particles
    particles = make_particles(num_particles)

    # Initialize visualization
    window = init_graphics(particles)
    clock = pygame.time.Clock()

    # Perform simulation
    start = time()
    for step in range(steps):
        # Compute forces
        # for p1 in particles:
        #     p1.ax = p1.ay = 0 # reset accleration to 0
        #     for p2 in particles:
        #         p1.apply_force(p2)
        window.fill(WHITE)
        for p in particles:
            p.draw(window, Particle.radius)

        # compute forces


        # Move particles
        for p in particles:
            p.move()
            
        
        pygame.display.flip()
    

    end = time()

    print('serial simulation took {0} seconds'.format(end - start))
    

##########################
# MAIN #
##########################

def run():
    serial_simulation(num_particles, num_steps, 1)

run()
