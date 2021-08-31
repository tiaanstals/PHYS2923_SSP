###########################
# Variables and Imports #
###########################
from random import random, seed, shuffle, uniform
from math import ceil, sqrt, atan2, cos, inf, pi, sin
from functools import reduce
from time import time
import pygame
from pygame import gfxdraw
import numpy as np

#A comment
num_particles = 4
num_steps = 10000
scale = 1e+8
x_lim = 1e-9
y_lim = 1e-9
x_lim = x_lim*scale
y_lim = y_lim*scale
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
DT = 0.005




###########################
# Graphics #
###########################
BLUE = [0, 0, 255]
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
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

    radius = radius*scale

    int_radius = int(ceil(radius/x_lim* WINDOW_HEIGHT))

    dt = DT


    def __init__(self, x, y, vx, vy, ax, ay, id):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay
        self.oldx = None
        self.oldy = None
        self.new_ax = 0
        self.new_ay = 0
        self.id = id
        self.collision = False
        self.oldx = x
        self.oldy = y

    def draw(self, window, rad):
            """
            Create a graphical representation of this particle for visualization.
            win is the graphics windown in which the particle should be drawn.
            
            rad is the radius of the particle"""
            x_coord = int(self.x/x_lim * WINDOW_WIDTH)
            y_coord = int(self.y/y_lim * WINDOW_HEIGHT)
            # if abs(x_coord) > WINDOW_WIDTH or abs(y_coord) > WINDOW_HEIGHT or x_coord < 0 or y_coord < 0:
            #     x_coord = x_lim/2 * WINDOW_WIDTH
            #     y_coord = y_lim/2 * WINDOW_WIDTH
            #     return
            draw_circle(window, x_coord, y_coord, Particle.int_radius, BLUE)

    def move(self):
        """ if a particle collides, it cannot continue with the last vx/vy components
        Therefore we must update the vx/vy components using the new acceleration, and then move
        This still has some issues, such as a particle might not move enough to get outside the 
        radius of another particle
        """
        
        if self.collision:
            self.vx = self.vx + (self.ax + self.new_ax)/2 * self.dt
            self.vy = self.vy + (self.ay + self.new_ay)/2 * self.dt
            self.x = self.oldx + self.vx * self.dt + 0.5*self.ax*self.dt**2
            self.y = self.oldy + self.vy * self.dt + 0.5*self.ay*self.dt**2
            self.ax = self.new_ax
            self.ay = self.new_ay
        else:
            self.x = self.x + self.vx * self.dt + 0.5*self.ax*self.dt**2
            self.y = self.y + self.vy * self.dt + 0.5*self.ay*self.dt**2
            self.vx = self.vx + (self.ax + self.new_ax)/2 * self.dt
            self.vy = self.vy + (self.ay + self.new_ay)/2 * self.dt
            self.ax = self.new_ax
            self.ay = self.new_ay
        self.oldx = self.x
        self.oldy = self.y
    
    def print(self):
        print("x: " + str(self.x))
        print("y: " + str(self.y))
        print("vx: " + str(self.vx))
        print("vy: " + str(self.vy))
        print("ax: " + str(self.ax))
        print("ay: " + str(self.ay))
        print("new_ax: " + str(self.new_ax))
        print("new_ay: " + str(self.new_ay))
    
    def compute_wall_forces(self):
        BOTTOM_LEFT = (0, 0)
        BOTTOM_RIGHT = (x_lim, 0)
        TOP_LEFT = (0, y_lim)
        TOP_RIGHT = (x_lim, y_lim)
        
        horizontal_count = 0
        vertical_count = 0

        #compute distances
        d_left_wall = distance_point_to_wall(BOTTOM_LEFT, TOP_LEFT, self.x, self.y)
        d_right_wall = distance_point_to_wall(BOTTOM_RIGHT, TOP_RIGHT, self.x, self.y)
        d_bottom_wall = distance_point_to_wall(BOTTOM_LEFT, BOTTOM_RIGHT, self.x, self.y)
        d_top_wall = distance_point_to_wall(TOP_LEFT, TOP_RIGHT, self.x, self.y)
        # print("d_left_wall" + str(d_left_wall))
        # print("d_right_wall" + str(d_right_wall))
        # print("d_bottom_wall" + str(d_bottom_wall))
        # print("d_top_wall" + str(d_top_wall))

        # check if any particles satisfies close condition. Note they must be heading in the right direction too!

        
        left_wall = closer_than_radius(d_left_wall)
        if left_wall == 1 and self.vx > 0:
            left_wall = 0
        
        right_wall = closer_than_radius(d_right_wall)
        if right_wall == 1 and self.vx < 0:
            right_wall = 0

        top_wall = closer_than_radius(d_top_wall)
        if top_wall == 1 and self.vy < 0:
            top_wall = 0
        bottom_wall = closer_than_radius(d_bottom_wall)
        if bottom_wall == 1 and self.vy > 0:
            bottom_wall = 0
            
        # print("left_wall" + str(left_wall))
        # print("right_wall" + str(right_wall))
        # print("top_wall" + str(top_wall))
        # print("bottom_wall" + str(bottom_wall))
        

        ## First check horizontal walls
        vertical_count = left_wall + right_wall
        horizontal_count = bottom_wall + top_wall
        # print("vertical_count" + str(vertical_count))
        # print("horizontal_count" + str(horizontal_count))
        # corner
        if vertical_count + horizontal_count > 1:
            # here we need to compute new vx and vy and convert to ax and ay
            self.new_ax = -2*self.vx/Particle.dt
            self.new_ay = -2*self.vy/Particle.dt
            return
        # vertical wall
        elif vertical_count > 0:
            self.new_ax = -2*self.vx/Particle.dt
        #horizontal wall
        elif horizontal_count > 0:
            self.new_ay = -2*self.vy/Particle.dt
            self.print()
        return

    def compute_elastic_collision_forces(self, other_p):
        """computes the forces for an elastic collsion
        :param other_p: object of type Particle
        :return: None
        """

        # first check if particles are within range
        # if they collided previous round do nothing
        if two_particles_bounce(self, other_p):
            # https://en.wikipedia.org/wiki/Elastic_collision
            print(self.id)
            v1 = np.array([self.vx, self.vy])
            print("V1==============V1")
            print("V1 ID" + str(self.id))
            print("Before" + str(v1))
            
            v2 = np.array([other_p.vx, other_p.vy])
            print("V2==============V2")
            print("Before" + str(v2))

            # note here we are doing p2-p1
            
            normal_vector = np.array([other_p.x-self.x,other_p.y-self.y])
            print("NORMAL==============NORMAL")
            print(str(normal_vector))

            
            unit_normal = normal_vector/sqrt(normal_vector[0]**2 + normal_vector[1]**2)
            print("unit_normal==============unit_normal")
            print(str(unit_normal))


            unit_tangent = np.array([-unit_normal[1], unit_normal[0]])
            print("unit_tangent==============unit_tangent")
            print(str(unit_tangent))

            # negative s
            # ign since we want component of velocity facing towards the particle
            v1_normal = unit_normal @ v1
            v1_tangent = unit_tangent @ v1
            print("v1_normal==============v1_normal")
            print(str(v1_normal))
            print("v1_tangent==============v1_tangent")
            print(str(v1_tangent))

            v2_normal = unit_normal @ v2
            v2_tangent = unit_tangent @ v2
            print("v2_normal==============v2_normal")
            print(str(v2_normal))
            print("v2_tangent==============v2_tangent")
            print(str(v2_tangent))

            v_1_normal_hold = v1_normal
            v1_normal = v2_normal
            v2_normal = v_1_normal_hold
            print("UPDATED v1_normal============== UPDATED v1_normal")
            print(str(v1_normal))
            print("UPDATED v2_normal============== UPDATED v2_normal")
            print(str(v2_normal))

            v1_normal_vec = v1_normal*(unit_normal)
            v1_tangent_vec = v1_tangent*unit_tangent

            v2_normal_vec = v2_normal*unit_normal
            v2_tangent_vec = v2_tangent*unit_tangent

            v1_final = v1_normal_vec + v1_tangent_vec
            v2_final = v2_normal_vec + v2_tangent_vec

            
            
            print("After" + str(v1_final))
            print()
            v1_change = v1_final - v1
            v2_change = v2_final- v2
            
            i_hat = np.array([1,0])
            j_hat = np.array([0,1])

            self.new_ax += (v1_change @ i_hat)/Particle.dt
            self.new_ay += (v1_change @ j_hat)/Particle.dt
            # self.print()

            other_p.new_ax += (v2_change @ i_hat)/Particle.dt
            other_p.new_ay += (v2_change @ j_hat)/Particle.dt
            self.collision = True
            other_p.collision = True
        return

                # self.move_after_collision(other_p,v1_final, v2_final)
    
###########################
# MATH FUNCTIONS #
###########################
def distance_point_to_wall(WALL_START, WALL_END, x, y):
    wall_start_x, wall_start_y = WALL_START
    wall_end_x, wall_end_y = WALL_END
     
    # avoid square root on the bottom (expensive operation) by squaring the top and squaring in closer_than_radius_function
    d = abs((wall_end_x - wall_start_x)*(wall_start_y-y)-(wall_start_x-x)*(wall_end_y-wall_start_y))**2/((wall_end_x-wall_start_x)**2 + (wall_end_y-wall_start_y)**2)
    return d

radius_dist = (Particle.radius)**2
def closer_than_radius(distance):
    if distance <= radius_dist:
        # print("closer")
        return 1
    else:
        # print("further")
        return 0

collision_distance = (2*Particle.radius)**2
def two_particles_bounce(p1, p2):
    if (p2.x-p1.x) < radius_dist:
        if (p2.x-p1.x)**2 + (p2.y-p1.y)**2 <= collision_distance:
            return True
    else:
        return False

##################
# Initialization #
##################
def make_particles(n):
    """Construct a list of n particles in two dimensions, initially distributed
    evenly but with random velocities. The resulting list is not spatially
    sorted."""
    seed(1000)
    particles = [Particle(0, 0, 0, 0, 0, 0, _) for _ in range(n)]

    # Make sure particles are not spatially sorted
    shuffle(particles)

    for p in particles:
        # Distribute particles randomly in our box
        x_coord = uniform(0, x_lim)
        y_coord = uniform(0, y_lim)
        p.x = uniform(0, x_lim)
        p.y = uniform(0, y_lim)

        # Assign random velocities within a bound
        temp_scale = 1
        p.vx = temp_scale*uniform(-x_lim, x_lim)
        p.vy = temp_scale*uniform(-y_lim, y_lim)

        p.ax = 0
        p.ay = 0

    return particles

def init_graphics(particles):
    background_colour = WHITE
    pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
    
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('Particle Simulation')
    
    window.fill(background_colour)
    return window

    



#####################
# Serial Simulation #
#####################
def serial_simulation(n, steps, update_interval=1, label_particles=False):

    # Create particles
    particles = make_particles(num_particles)

    # Initialize visualization
    window = init_graphics(particles)
    clock = pygame.time.Clock()
    myfont = pygame.font.Font('Roboto-Medium.ttf', 10)
    
    # Perform simulation
    start = time()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    pause()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        window.fill(WHITE)
        for p in particles:
            p.draw(window, Particle.radius)
            if label_particles:
                label = myfont.render(str(p.id), 2, BLACK )
                window.blit(label, (p.x/x_lim*WINDOW_WIDTH, p.y/y_lim*WINDOW_HEIGHT))

        #compute forces
        for p in particles:
            p.new_ax = 0
            p.new_ay = 0
            p.compute_wall_forces() 

            #compute collision interactions
            for p2 in particles:
                if p.id == p2.id:
                    continue
                else:
                    p.compute_elastic_collision_forces(p2)
                

        # Move particles
        for p in particles:
            p.move()
            p.collision = False

        bottom_left_corner = myfont.render("0,0", 1, BLUE)
        window.blit(bottom_left_corner, (0, 0))
        top_left_corner = myfont.render("0,1", 1, BLUE)
        window.blit(top_left_corner, (0, WINDOW_HEIGHT-15))     
        bottom_right_corner = myfont.render("1,0", 1, BLUE)
        window.blit(bottom_right_corner, (WINDOW_WIDTH-15, 0))    
        top_right_corner = myfont.render("1,1", 1, BLUE)
        window.blit(top_right_corner, (WINDOW_WIDTH-15, WINDOW_HEIGHT-15))   

        pygame.display.flip()
    

    end = time()

    print('serial simulation took {0} seconds'.format(end - start))

def pause():
    paused = True

    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    paused == False
                    return
                elif event.key == pygame.K_q:
                    pygame.quit()
        
##########################
# MAIN #
##########################

def run():
    max_particle_speed = Particle.dt * Particle.radius
    print("max_particle_speed" + str(max_particle_speed))
    serial_simulation(num_particles, num_steps, 1, True)

# d= distance_point_to_wall((0,0),(10,0),10,10)
# print(d)
run()