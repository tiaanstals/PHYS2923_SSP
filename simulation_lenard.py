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
import pygame_widgets
from pygame_widgets.slider import Slider

num_particles = 50
num_steps = 10000
scale = 1e+9
x_lim = 3e-9 
y_lim = 3e-9
x_lim = x_lim*scale
y_lim = y_lim*scale
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
CONTROL_SPACE = 100
SPACING_TEXT = 15
DT = 0.001
radius = 5.29177210903e-11
scale_factor = 10





###########################
# Graphics #
###########################
BLUE = [0, 0, 255]
WHITE = [255, 255, 255]
RED = [255,0,0]
BLACK = [0, 0, 0]
def draw_circle(window, x, y, radius, color):
    # Draw anti-aliased circle
    gfxdraw.aacircle(window, x, y, radius, color)
    gfxdraw.filled_circle(window, x, y, radius, color)

def init_graphics(particles):
    background_colour = WHITE
    pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
    
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT + CONTROL_SPACE))
    pygame.display.set_caption('Particle Simulation')
    
    window.fill(background_colour)
    return window

def draw_line(window, coord_1, coord_2):
    pygame.draw.line(window, BLACK, coord_1, coord_2)
            

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
    print(int_radius)

    dt = DT

    energy_correction = 1


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
        self.constant_particle = False

    def draw(self, window, rad):
            """
            Create a graphical representation of this particle for visualization.
            win is the graphics windown in which the particle should be drawn.
            
            rad is the radius of the particle"""
            x_coord = int(self.x/x_lim * WINDOW_WIDTH)
            y_coord = int(self.y/y_lim * WINDOW_HEIGHT)
            # print("x_coord " + str(x_coord))
            # print("y_coord"+str(y_coord))
            # self.print()
            # if abs(x_coord) > WINDOW_WIDTH or abs(y_coord) > WINDOW_HEIGHT or x_coord < 0 or y_coord < 0:
            #     x_coord = x_lim/2 * WINDOW_WIDTH
            #     y_coord = y_lim/2 * WINDOW_WIDTH
            #     return
            # if (x_coord < 0) or (y_coord < 0) or (x_coord > WINDOW_WIDTH) or (y_coord > WINDOW_HEIGHT):
            #     return
            color = BLUE
            if self.constant_particle:
                color = RED
            draw_circle(window, x_coord, y_coord, Particle.int_radius, color)

    def move(self):
        """ if a particle collides, it cannot continue with the last vx/vy components
        Therefore we must update the vx/vy components using the new acceleration, and then move
        This still has some issues, such as a particle might not move enough to get outside the 
        radius of another particle
        """
        if not self.constant_particle:
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
        energy_before = self.energy
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
        
        return

    def compute_lj_forces(self, other_p):
        """
        Calculate the acceleration on each 
        particle as a  result of each other 
        particle. 
        """ 
        # calculate x and y distances between the two points
        rx = self.x - other_p.x
        ry = self.y - other_p.y
        r2 = sqrt(rx*rx+ry*ry)

        lj_f = lj_force(r2)
        print("ljforce: " + str(lj_f))
        energy_before = self.energy + other_p.energy

        self.new_ax += (lj_f*(rx/r2))/self.mass
        self.new_ay += (lj_f*(ry/r2))/self.mass
        other_p.new_ax -= (lj_f*(rx/r2))/self.mass
        other_p.new_ay -= (lj_f*(ry/r2))/self.mass
        energy_after = self.energy_plus_dt + other_p.energy_plus_dt
        ratio = energy_before/energy_after
        if 1 - ratio > 0.1:
            self.new_ax *= ratio
            self.new_ay *= ratio 
            other_p.new_ax *= ratio
            other_p.new_ax *= ratio
        return
    
    def change_temp(self, change):
        energy_before = self.energy
        self.vx = self.vx + self.vx*change
        self.vy = self.vx + self.vx*change
        return self.energy - energy_before
    
    @property
    def energy(self):
        """Return the kinetic energy of this particle."""
        return 0.5 * self.mass * (self.vx ** 2 + self.vy ** 2)
    
    @property 
    def energy_plus_dt(self):
        """Return the kinetic energy in 1 step"""
        new_vx = self.vx + (self.ax + self.new_ax)/2 * self.dt
        new_vy = self.vy + (self.ay + self.new_ay)/2 * self.dt
        return 0.5 * self.mass * (new_vx ** 2 + new_vy ** 2)

    
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

def lj_force(r, epsilon=0.09, sigma=2.45):
    """
    Implementation of the Lennard-Jones potential 
    to calculate the force of the interaction.

    The force scalar is the derivative wrt to R 
    f = -partial(LJ_potential)/partial*r
    """
    r = 1/Particle.radius * r
    
    if r < 2.1:
        print(r)
        r = 1.99
    return 48 * epsilon * (pow(sigma,12)/pow(r,13)-24*epsilon*pow(sigma,6)/pow(r,7))


##################
# Initialization #
##################
def make_particles(n, temp_scale, nucleation):
    """Construct a list of n particles in two dimensions, initially distributed
    evenly but with random velocities. The resulting list is not spatially
    sorted."""
    seed(1000)
    particles = [Particle(0, 0, 0, 0, 0, 0, _) for _ in range(n)]

    # Make sure particles are not spatially sorted
    shuffle(particles)

    for p in particles:
        # Distribute particles randomly in our box
        p.x = uniform(0+Particle.radius, x_lim-Particle.radius)
        p.y = uniform(0+Particle.radius, y_lim-Particle.radius)

        # Assign random velocities within a bound
        p.vx = temp_scale*uniform(-x_lim, x_lim)
        p.vy = temp_scale*uniform(-y_lim, y_lim)

        p.ax = 0
        p.ay = 0
    
    if nucleation:
        new_particle = Particle(x_lim/2, y_lim/2,0,0,0,0,-1)
        new_particle.constant_particle = True
        particles.append(new_particle)
    # make sure no particles are overlapping
    overlap = True
    while overlap:
        collision_detected = False
        for i, p in enumerate(particles):
            for p2 in particles[i+1:]:
                if sqrt((p.x-p2.x)**2 + (p.y-p2.y)**2) <= 2*Particle.radius:
                    if p.constant_particle:
                        p2.x = uniform(0+Particle.radius, x_lim-Particle.radius)
                        p2.y = uniform(0+Particle.radius, x_lim-Particle.radius)
                    else:
                        p.x = uniform(0+Particle.radius, x_lim-Particle.radius)
                        p.y = uniform(0+Particle.radius, x_lim-Particle.radius)
                    collision_detected = True
        if not collision_detected:
            overlap = False
                
    return particles



#####################
# Serial Simulation #
#####################
def serial_simulation(n, steps, update_interval=1, label_particles=False, normalize_energy=True, nucleation=False):

    # Create particles
    temp_scale = 0.3
    particles = make_particles(num_particles, temp_scale, nucleation)
    initial_energy = reduce(lambda x, p: x + p.energy, particles, 0)

    # Initialize visualization
    window = init_graphics(particles)
    clock = pygame.time.Clock()
    myfont = pygame.font.Font('Roboto-Medium.ttf', 10)
    # https://pygamewidgets.readthedocs.io/
    # xcoord, ycoord, width, height, min, max
    
    slider = Slider(window, 0, WINDOW_HEIGHT + SPACING_TEXT, WINDOW_WIDTH, 10, min=0.001, max=3, step=.05, initial=temp_scale)
    # Perform simulation
    start = time()
    running = True
    energy_display = initial_energy

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
        draw_line(window, (0,WINDOW_HEIGHT), (WINDOW_WIDTH, WINDOW_HEIGHT))
        slider_val = round(slider.getValue(),3)
        label = myfont.render("Temperature: " + str(round(energy_display*100,3)), 2, BLACK )
        window.blit(label, (1, WINDOW_HEIGHT + 1))
        pygame_widgets.update(event)
        for p in particles:
            p.draw(window, Particle.radius)
            if label_particles:
                label = myfont.render(str(p.id), 2, BLACK )
                window.blit(label, (p.x/x_lim*WINDOW_WIDTH, p.y/y_lim*WINDOW_HEIGHT))

        #compute forces
        for i, p in enumerate(particles):
            p.new_ax = 0
            p.new_ay = 0
            p.compute_wall_forces() 

            #compute collision interactions
            for p2 in particles[i+1:]:
                p.compute_lj_forces(p2)
                
        energy = 0
        change_temp_scale = (slider_val - temp_scale)
        
        change_energy = 0
        # Move particles
        for p in particles:
            p.move()
            # energy normalisation
            p.vx *= Particle.energy_correction
            p.vy *= Particle.energy_correction
            energy += p.energy
            # change temp scale
            if change_temp_scale != 0:
                change_energy += p.change_temp(change_temp_scale)
            
            p.collision = False
        initial_energy += change_energy
        # need to change initial energy
        temp_scale = slider_val
        # Energy normalization
        if normalize_energy:
            Particle.energy_correction = sqrt((initial_energy) / energy)
        energy_display = energy*Particle.energy_correction
        bottom_left_corner = myfont.render("0,0", 1, BLUE)
        window.blit(bottom_left_corner, (0, 0))
        top_left_corner = myfont.render("0,1", 1, BLUE)
        window.blit(top_left_corner, (0, WINDOW_HEIGHT-15))     
        bottom_right_corner = myfont.render("1,0", 1, BLUE)
        window.blit(bottom_right_corner, (WINDOW_WIDTH-15, 0))    
        top_right_corner = myfont.render("1,1", 1, BLUE)
        window.blit(top_right_corner, (WINDOW_WIDTH-15, WINDOW_HEIGHT-15))   

        pygame.display.update()


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
    serial_simulation(num_particles, num_steps, 1, True, nucleation=True)
    

# d= distance_point_to_wall((0,0),(10,0),10,10)
# print(d)
run()