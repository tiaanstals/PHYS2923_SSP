###########################
# Variables and Imports #
###########################
from random import random, seed, shuffle, uniform
from math import ceil, sqrt, atan2, cos, inf, pi, sin, isclose, floor
from functools import reduce
from time import time
import pygame
from pygame import gfxdraw
import numpy as np
import pygame_widgets
from pygame_widgets.slider import Slider
from tqdm import tqdm
import pickle
import os


def lj_force_cutoff(r, SIGMA, EPSILON, SIGMA_6):
    """
    Lenard Jones function to calculate force at cutoff so we can have a vertical shift
    such the LJ force goes to 0 smnoothly at the cutoff
    """    
    # power functions are computationally expensive
    # Better to do multiplication 
    # https://drive.google.com/drive/folders/1ZcSIopqmLPnpFK4UUaBi6hvoybtvNeOs?usp=sharing
    inv_r = 1/r
    r_squared = r*r
    inv_r_squared = 1/r_squared
    inv_r_6 = inv_r_squared*inv_r_squared*inv_r_squared
    # constant out front is 24* epsilon/sigma
    const = 24*EPSILON/SIGMA
    # by convention repuslive force is positve and attractive force is negative
    # repulsive force is 2*(sigma/r)^13
    repulsive = 2*inv_r_6*inv_r_6*inv_r*SIGMA_6*SIGMA_6*SIGMA
    # attractive force is -(sigma/r)^6
    attractive = -inv_r_6*inv_r*SIGMA_6*SIGMA
    #final force F(r)=-du(r)/dr=24*(epsilon/sigma)*(2*(sigma/r)^13 - (sigma/r)^6) = constant*(repulsive + attractive)
    lf = const*(repulsive + attractive)
    return lf


class Simulation(object):
    num_particles = 20
    num_steps = 6000
    velocity_scaler = 1
    lim = 10
    x_lim, y_lim = lim, lim
    WINDOW_WIDTH = 600
    WINDOW_HEIGHT = 600
    CONTROL_SPACE = 100
    SPACING_TEXT = 15
    DT = 0.001
    EPSILON=3
    SIGMA=1
    CUTOFF = 2.5*SIGMA
    SIGMA_6 = SIGMA*SIGMA*SIGMA*SIGMA*SIGMA*SIGMA
    LONG_RANGE_POTENTIAL_CORRECTION=-8*np.pi*(num_particles/(x_lim*y_lim))/(3*CUTOFF**3)
    # LJ_FORCE_SHIFT=-lj_force_cutoff(CUTOFF, SIGMA, EPSILON, SIGMA_6)

def set_sim_params(params, simulation):
    simulation.num_particles = params.num_particles
    simulation.num_steps = params.num_steps
    simulation.velocity_scaler = params.velocity_scaler
    simulation.lim = params.lim
    simulation.x_lim, simulation.y_lim = params.x_lim, params.y_lim
    simulation.WINDOW_WIDTH = params.WINDOW_WIDTH 
    simulation.WINDOW_HEIGHT = params.WINDOW_HEIGHT
    simulation.CONTROL_SPACE = params.CONTROL_SPACE
    simulation.SPACING_TEXT = params.SPACING_TEXT
    simulation.DT = params.DT
    simulation.EPSILON=params.EPSILON
    simulation.SIGMA=params.SIGMA
    simulation.CUTOFF = params.CUTOFF
    simulation.SIGMA_6 = params.SIGMA_6
    # simulation.LJ_FORCE_SHIFT= params.LJ_FORCE_SHIFT
    simulation.LONG_RANGE_POTENTIAL_CORRECTION= params.LONG_RANGE_POTENTIAL_CORRECTION

###########################
# Graphics #
###########################
BLUE = [0, 0, 255]
WHITE = [255, 255, 255]
RED = [255,0,0]
BLACK = [0, 0, 0]
LIGHT_BLUE = [173,216,230]
def draw_circle(window, x, y, radius, color_inner, color_outer):
    # Draw circle (ellipse with float radius)
    # second argument is a rect
    # https://www.pygame.org/docs/ref/rect.html#pygame.Rect
    # (left, top, width, height)
    pygame.draw.ellipse(window, color_inner, (x-radius/2, y-radius/2, radius,radius))
    

def init_graphics():
    background_colour = WHITE
    pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
    
    window = pygame.display.set_mode((Simulation.WINDOW_WIDTH, Simulation.WINDOW_HEIGHT + Simulation.CONTROL_SPACE))
    pygame.display.set_caption('Particle Simulation')
    
    window.fill(background_colour)
    return window

def draw_line(window, coord_1, coord_2):
    pygame.draw.line(window, BLACK, coord_1, coord_2)
            
def draw_particle(window, rad, x, y, color_inner, color_outer, lattice_particle):
            """
            Create a graphical representation of this particle for visualization.
            win is the graphics windown in which the particle should be drawn.
            
            rad is the radius of the particle"""
            if lattice_particle:
                x = int(to_display_scale(x))
                y = int(to_display_scale(y))
                # print("[" + str(x) + "," + str(y) + "]")
                # print(Particle.display_radius)
                draw_circle(window, x, y, 1, BLACK, color_outer)
            x = int(to_display_scale(x))
            y = int(to_display_scale(y))
            # print("[" + str(x) + "," + str(y) + "]")
            # print(Particle.display_radius)
            draw_circle(window, x, y, Particle.display_radius, color_inner, color_outer)

def to_display_scale(num):
    return num/Simulation.x_lim * Simulation.WINDOW_HEIGHT
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
    

    scale_pos = None

    radius = 1
    mass = 1

    display_radius = to_display_scale(radius)

    dt = Simulation.DT

    energy_correction = 1
    epsilon=Simulation.EPSILON
    sigma=Simulation.SIGMA


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
        self.potential_energy = 0
        self.force_on_wall = 0
        self.lattice_position = False
        self.dont_move_overlap = False
    
    def set_new_acc(self, new_ax, new_ay):
        self.ax += new_ax
        self.ay += new_ay

    def set_potential_energy(self,u):
        self.potential_energy += u

    def move(self):
        """ if a particle collides, it cannot continue with the last vx/vy components
        Therefore we must update the vx/vy components using the new acceleration, and then move
        This still has some issues, such as a particle might not move enough to get outside the 
        radius of another particle
        https://www.ccp5.ac.uk/sites/www.ccp5.ac.uk/files/Democritus/Theory/verlet.html#velver
        """
        if (not self.constant_particle) and (not self.lattice_position):
            vx_half = self.vx + (self.ax*self.dt)/2
            vy_half = self.vy + (self.ay*self.dt)/2

            self.x = self.x + vx_half * self.dt
            self.y = self.y + vy_half * self.dt 

            self.vx = vx_half + (self.dt*self.new_ax)/2
            self.vy = vy_half + (self.dt*self.new_ay)/2
            self.ax = self.new_ax
            self.ay = self.new_ay

    
    def print(self):
        print("id: " + str(self.id))
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
        BOTTOM_RIGHT = (Simulation.x_lim, 0)
        TOP_LEFT = (0, Simulation.y_lim)
        TOP_RIGHT = (Simulation.x_lim,Simulation.y_lim)
        
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
            self.force_on_wall += abs(2*self.vx/Particle.dt)
            self.force_on_wall += abs(2*self.vy/Particle.dt)
        # vertical wall
        elif vertical_count > 0:
            self.new_ax = -2*self.vx/Particle.dt
            self.force_on_wall += abs(2*self.vx/Particle.dt)
        #horizontal wall
        elif horizontal_count > 0:
            self.new_ay = -2*self.vy/Particle.dt
            self.force_on_wall += abs(2*self.vy/Particle.dt)
        
        energy_after = self.energy_plus_dt
        if isclose(energy_before,0):
            return


        return

    def compute_lj_forces(self, other_p, record_potential):
        """
        Calculate the acceleration on each 
        particle as a  result of each other 
        particle. 
        """ 
        # calculate x and y distances between the two points
        # vector points from other_p to current particle
        if self.lattice_position or other_p.lattice_position:
            return -1
        rx = self.x - other_p.x
        ry = self.y - other_p.y
        r2 = sqrt(rx*rx+ry*ry)

        lj_f = lj_force(r2)
        if record_potential:
            lj_u = lj_potential(r2)
            self.potential_energy += lj_u
            other_p.set_potential_energy(lj_u)
        if isclose(lj_f, 0):
            return r2
        # old_ax = self.new_ax
        energy_before = self.energy + other_p.energy
        # if lj_f is positive, acceleration should be in the direction of the vector which points from p2 to p1 (repulsive)
        # if lj_f is negative, acceleration should be in the direction of the vector which points from p1 to p2 (attractive)
        self.set_new_acc(lj_f*rx/r2,lj_f*ry/r2)
        # by newtons third law
        other_p.set_new_acc(-lj_f*rx/r2,-lj_f*ry/r2)
        return r2
    
    def compute_lattice_forces(self, other_p):
        rx = self.x - other_p.x
        ry = self.y - other_p.y
        r2 = sqrt(rx*rx+ry*ry)

        lj_f = lj_force_attractive_only(r2)
        if isclose(lj_f, 0):
            return r2
        # old_ax = self.new_ax

        # if lj_f is positive, acceleration should be in the direction of the vector which points from p2 to p1 (repulsive)
        # if lj_f is negative, acceleration should be in the direction of the vector which points from p1 to p2 (attractive)

        if isclose(r2, 0):
            return 0
        self.set_new_acc(lj_f*rx/r2,lj_f*ry/r2)
        # by newtons third law
        other_p.set_new_acc(-lj_f*rx/r2,-lj_f*ry/r2)
        return r2
    
    def change_temp(self, current_temp, new_temp):
        energy_before = self.energy
        self.vx = self.vx*sqrt(new_temp/current_temp)
        self.vy = self.vy*sqrt(new_temp/current_temp)
        return self.energy - energy_before
    
    @property
    def energy(self):
        """Return the kinetic energy of this particle."""
        return 0.5 * (self.vx ** 2 + self.vy ** 2)
    
    @property 
    def energy_plus_dt(self):
        """Return the kinetic energy in 1 step"""
        vx_half = self.vx + (self.ax*self.dt)/2
        vy_half = self.vy + (self.ay*self.dt)/2
        vx = vx_half + (self.dt*self.new_ax)/2
        vy = vy_half + (self.dt*self.new_ay)/2
        return 0.5 * (vx ** 2 + vy ** 2)

    @property 
    def total_velocity(self):
        return sqrt(self.vx**2+self.vy**2)

    
###########################
# MATH FUNCTIONS #
###########################
def distance_point_to_wall(WALL_START, WALL_END, x, y):
    wall_start_x, wall_start_y = WALL_START
    wall_end_x, wall_end_y = WALL_END
     
    # avoid square root on the bottom (expensive operation) by squaring the top and squaring in closer_than_radius_function
    d = abs((wall_end_x - wall_start_x)*(wall_start_y-y)-(wall_start_x-x)*(wall_end_y-wall_start_y))**2/((wall_end_x-wall_start_x)**2 + (wall_end_y-wall_start_y)**2)
    return d

def closer_than_radius(distance):
    if distance <= Particle.radius:
        # print("closer")
        return 1
    else:
        # print("further")
        return 0

def lj_force(r):
    """
    Implementation of the Lennard-Jones potential 
    to calculate the force of the interaction.

    The force scalar is the derivative wrt to R 
    f = -partial(LJ_potential)/partial*r
    """    
    if r > Simulation.CUTOFF:
        return 0

    # power functions are computationally expensive
    # Better to do multiplication 
    # https://drive.google.com/drive/folders/1ZcSIopqmLPnpFK4UUaBi6hvoybtvNeOs?usp=sharing
    inv_r = 1/r
    r_squared = r*r
    inv_r_squared = 1/r_squared
    inv_r_6 = inv_r_squared*inv_r_squared*inv_r_squared
    # constant out front is 24* epsilon/sigma
    const = 24*Simulation.EPSILON/Simulation.SIGMA
    # by convention repuslive force is positve and attractive force is negative
    # repulsive force is 2*(sigma/r)^13
    repulsive = 2*inv_r_6*inv_r_6*inv_r*Simulation.SIGMA_6*Simulation.SIGMA_6*Simulation.SIGMA
    # attractive force is -(sigma/r)^6
    attractive = -inv_r_6*inv_r*Simulation.SIGMA_6*Simulation.SIGMA
    #final force F(r)=-du(r)/dr=24*(epsilon/sigma)*(2*(sigma/r)^13 - (sigma/r)^6) = constant*(repulsive + attractive) + force_shift
    lf = const*(repulsive + attractive)
    return lf

def lj_force_attractive_only(r):
    """
    Implementation of the Lennard-Jones potential 
    to calculate the force of the interaction.

    The force scalar is the derivative wrt to R 
    f = -partial(LJ_potential)/partial*r
    """    
    if r > Simulation.CUTOFF:
        return 0

    # power functions are computationally expensive
    # Better to do multiplication 
    # https://drive.google.com/drive/folders/1ZcSIopqmLPnpFK4UUaBi6hvoybtvNeOs?usp=sharing

    #need to shift slightly to avoid division by zero
    r = r + 1.4
    inv_r = 1/r
    r_squared = r*r
    inv_r_squared = 1/r_squared
    inv_r_6 = inv_r_squared*inv_r_squared*inv_r_squared
    # constant out front is 24* epsilon/sigma
    const = 5.5*24*Simulation.EPSILON/Simulation.SIGMA
    # by convention repuslive force is positve and attractive force is negative
    # attractive force is -(sigma/r)^6
    attractive = -inv_r_6*inv_r*Simulation.SIGMA_6*Simulation.SIGMA
    #final force F(r)=-du(r)/dr=24*(epsilon/sigma)*(2*(sigma/r)^13 - (sigma/r)^6) = constant*(repulsive + attractive) + force_shift
    lf = const*(attractive)
    return lf

def lj_potential(r):
    """
    Implementation of the Lennard-Jones potential 
    used for calculating the potential energy of the particle during the interaction
    """
    inv_r = 1/r
    r_squared = r*r
    inv_r_squared = 1/r_squared
    inv_r_6 = inv_r_squared*inv_r_squared*inv_r_squared
    # constant out front is 4* epsilon
    const = 4*Simulation.EPSILON
    # by convention repuslive force is positve and attractive force is negative
    # repulsive force is (sigma/r)^12
    repulsive = inv_r_6*inv_r_6*Simulation.SIGMA_6*Simulation.SIGMA_6
    # attractive force is -(sigma/r)^6
    attractive = -inv_r_6*Simulation.SIGMA_6
    lu = const*(repulsive + attractive)
    return lu


def generate_square_matrix(n, x_mid, y_mid,nucleation):
    cube_2 = cube_root(2)*0.9
    first_x = x_mid - cube_2*n*Particle.radius
    first_y = y_mid - cube_2*n*Particle.radius
    x = []
    y = []
    for i in range(2*n+1):
        for j in range(2*n+1):
            x_coord = first_x + i*cube_2*Particle.radius 
            y_coord = first_y + j*cube_2*Particle.radius
            if isclose(x_mid, x_coord) and isclose(y_mid,y_coord) and nucleation:
                continue
            x.append(x_coord)
            y.append(y_coord)
    return [x,y]

def cube_root(x):
    if 0<=x: return x**(1./3.)
    return -(-x)**(1./3.)


def rotate_square_matrix(mat,deg):
    x_coords = mat[0]
    y_coords = mat[1]
    x_coords_rotated = []
    y_coords_rotated = []
    for i, x in enumerate(x_coords):
        y = y_coords[i]
        new_x = x*np.cos(deg) - y*np.sin(deg)
        new_y = x*np.sin(deg) + y*np.cos(deg)
        x_coords_rotated.append(new_x)
        y_coords_rotated.append(new_y)
    return [x_coords_rotated, y_coords_rotated]

##################
# Initialization #
##################
def make_particles(n, temp_scale, nucleation, locations, fast_particle=False, lattice_structure=False, lattice_size=2):
    """Construct a list of n particles in two dimensions, initially distributed
    evenly but with random velocities. The resulting list is not spatially
    sorted."""
    seed_num = int(random()*100)
    seed(seed_num)
    particles = [Particle(0, 0, 0, 0, 0, 0, _) for _ in range(n)]
    next_p = len(particles)
    # Make sure particles are not spatially sorted
    shuffle(particles)
    vx_list = []
    vy_list = []
    if locations is not None:
        for i, p in enumerate(particles):
            p.x = Simulation.x_lim/2 + locations[0][i]
            p.y = Simulation.y_lim/2 + locations[1][i]
            p.xy = 0
            p.vy = 0
            p.ax = 0
            p.ay = 0
            vx_list.append(p.vx)
            vy_list.append(p.vy)
    elif lattice_structure:
        # generates a 2d array with coordinates for the lattice
        locations = generate_square_matrix(lattice_size,0,0, nucleation)
        # locations = rotate_square_matrix(locations, 20)
        # first create lattice position particles
        particles = []
        for i in range(len(locations[0])):
            particles.append(Particle(0,0,0,0,0,0,i))
            # this property means the "particle" is actually just a preferred position
            particles[i].lattice_position = True
            particles[i].x = Simulation.x_lim/2 + locations[0][i]
            particles[i].y = Simulation.y_lim/2 + locations[1][i]
        next_p = i + 1
        for i in range(len(locations[0])):
            populate = random() < 0.6
            particle = Particle(0,0,0,0,0,0,next_p)
            particle.x = Simulation.x_lim/2 + locations[0][i]
            particle.y = Simulation.y_lim/2 + locations[1][i]
            particle.dont_move_overlap = True
            if populate:
                particles.append(particle)
                vx_list.append(particle.vx)
                vy_list.append(particle.vy)
                next_p += 1
        
        # add in some other particles
        # for i in range(round(len(locations[0]) * 0.5)):
        #     particle = Particle(0,0,0,0,0,0,next_p)
        #     particle.x = uniform(0+Particle.radius, Simulation.x_lim-Particle.radius)
        #     particle.y = uniform(0+Particle.radius, Simulation.y_lim-Particle.radius)
        #     particle.vx = temp_scale*uniform(-Simulation.x_lim, Simulation.x_lim)
        #     particle.vy = temp_scale*uniform(-Simulation.y_lim, Simulation.y_lim)
        #     particles.append(particle)
        #     vx_list.append(particle.vx)
        #     vy_list.append(particle.vy)
        #     next_p += 1                
        #     print(next_p)
    else:
        for p in particles:
            # Distribute particles randomly in our box
            p.x = uniform(0+Particle.radius, Simulation.x_lim-Particle.radius)
            p.y = uniform(0+Particle.radius, Simulation.y_lim-Particle.radius)

            # Assign random velocities within a bound
            p.vx = temp_scale*uniform(-Simulation.x_lim, Simulation.x_lim)
            p.vy = temp_scale*uniform(-Simulation.y_lim, Simulation.y_lim)
            vx_list.append(p.vx)
            vy_list.append(p.vy)
            p.ax = 0
            p.ay = 0
    
    # let total linear momentum be initially 0
    mean_vx = sum(vx_list)/len(vx_list)
    mean_vy = sum(vy_list)/len(vy_list)
    for p in particles:
        p.vx -= mean_vx
        p.vy -= mean_vy
    # make sure no particles are overlapping
    overlap = True
    if (locations is not None) and (lattice_structure == False):
        overlap = False
    while overlap:
        collision_detected = False
        print(overlap)
        for i, p in enumerate(particles):
            for p2 in particles[i+1:]:
                if round(sqrt((p.x-p2.x)**2 + (p.y-p2.y)**2),2) < (1.5*Particle.radius):
                    # print("Particle 1: [" + str(p.x) + "," + str(p.y) + "]")
                    # print("Particle 2: [" + str(p2.x) + "," + str(p2.y) + "]")
                    # print("(px - p2x)**2 = " + str((p.x-p2.x)**2))
                    # print("(py - p2y)**2 = " + str((p.y-p2.y)**2))
                    # if lattice structure, the ids of "free" particles is higher
                    if ((p.constant_particle) or (p.lattice_position) or (p.dont_move_overlap)) and ((p2.constant_particle) or (p2.lattice_position) or (p2.dont_move_overlap)):
                        collision_detected = False
                        continue
                    elif (p.constant_particle) or (p.lattice_position) or (p.dont_move_overlap):
                        p2.x = uniform(0+2*Particle.radius, Simulation.x_lim-2*Particle.radius)
                        p2.y = uniform(0+2*Particle.radius, Simulation.x_lim-2*Particle.radius)
                        collision_detected = True
                    else:
                        p.x = uniform(0+2*Particle.radius, Simulation.x_lim-2*Particle.radius)
                        p.y = uniform(0+2*Particle.radius, Simulation.x_lim-2*Particle.radius)
                        collision_detected = True
                    
        if not collision_detected:
            overlap = False
                
    return (particles, seed)



#####################
# Serial Simulation #
#####################
def serial_simulation(update_interval=1, label_particles=False, normalize_energy=True, nucleation=False, speed_up=5,load_data=False,
                        sim_name='latest', fast_particle=False, record_potential=False, lattice_structure=False, temp_change_start=1500,
                        temp_change_finish=3000, desired_temp=1.5):

    # Create particles
    if not load_data:

        # lattice structure
        locations = None
        # locations = None
        (particles,seed) = make_particles(Simulation.num_particles, Simulation.velocity_scaler, nucleation, locations, fast_particle, lattice_structure, lattice_size=2)
        Simulation.num_particles = len(particles)
        # Perform simulation
        start = time()
        running = True
        paths = np.zeros((Simulation.num_steps, len(particles), 6))
        rpaths_data = np.zeros((Simulation.num_steps, len(particles), len(particles)))
        colours = np.zeros(len(particles))
        lattice_data = np.zeros(len(particles))
        temperatures = np.zeros((Simulation.num_steps, 1))
        temperature = 0
        num_lattice_particles = 0
        # start changing temp after 1000 steps
        # change over 1000 steps
        temp_step_size = -1
        for step in tqdm(range(1,Simulation.num_steps)):
            #compute forces
            for i, p in enumerate(particles):
                paths[step][p.id][0] = p.x
                paths[step][p.id][1] = p.y
                # kinetic energy
                paths[step][p.id][2] = p.energy


                # potential energy
                if record_potential:
                    paths[step][p.id][3] = p.potential_energy

                # forces on walls (used for pressure calcs)
                #paths[step][p.id][4] = p.force_on_wall

                # total velocity (since m = 1, p = v)
                #paths[step][p.id][5] = p.total_velocity


                p.new_ax = 0
                p.new_ay = 0
                p.force_on_wall = 0
                p.potential_energy = 0

                #start changing temp after 1000 steps
                if step >= temp_change_start and step <= temp_change_finish:
                    p.change_temp(temperature, temperature + temp_step_size)

                # temp_change = 1
                # p.change_temp(0.00001)

                #compute collision interactions
                
                for p2 in particles:
                    if p2.id == p.id:
                        continue
                    r = p.compute_lj_forces(p2, record_potential)
                    # this means one particle is a lattice position
                    if r == -1:
                        r = p.compute_lattice_forces(p2)
                    # rpaths_data[step][p.id][p2.id] = r
                    # rpaths_data[step][p2.id][p.id] = r
                # particles in center of box
                p.compute_wall_forces() 
            
            # Move particles
            for p in particles:
                p.move()
            if step >= 500:
                if temp_step_size == -1:
                    current_temp = 1/(2*Simulation.num_particles) * sum(paths[step][:,2])
                    temp_step_size = (desired_temp - current_temp)/(temp_change_finish-temp_change_start)
            temperature = 1/(2*Simulation.num_particles) * sum(paths[step][:,2])
            temperatures[step] = temperature


    return (paths, Simulation, temperatures)

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
# find the closest grid location for each particle
def find_closest_grid_locs(locations, grid_locations, num_real_particles):
    closest_grid_locations = np.zeros(num_real_particles)
    indexes_atoms = np.zeros(num_real_particles)
    for j, location in enumerate(locations):
        closest = 400
        closest_index = -1
        for i, grid_point in enumerate(grid_locations):
            r2 = (location[0] - grid_point[0])**2 + (location[1] - grid_point[1])**2
            if r2 < closest:
                closest = r2
                closest_index = i
        closest_grid_locations[j] = closest_index
        indexes_atoms[j] = 25+j
    return (closest_grid_locations, indexes_atoms)


def find_indexes_of_sites_close(index_of_site,imaginary_grid_locations,dist_between_absorption_sites,dist_between_diagonal_absorption_sites):
    x = imaginary_grid_locations[index_of_site][0]
    y = imaginary_grid_locations[index_of_site][1]
    indexes_close = []
    indexes_diagonal = []
    for i, site in enumerate(imaginary_grid_locations):
        r2 = (x - site[0])**2 + (y-site[1])**2
        # make sure to not count the site itself
        if r2 < dist_between_absorption_sites*1.1 and r2 > dist_between_absorption_sites*0.9:
            indexes_close.append(i)
        if r2 < dist_between_diagonal_absorption_sites*1.1 and r2 > dist_between_diagonal_absorption_sites*0.9:
            indexes_diagonal.append(i)
    return (indexes_close, indexes_diagonal)

def main():
    pygame.init()
    print("x_lim {}".format(Simulation.x_lim))
    print("y_lim {}".format(Simulation.y_lim))
    print("Particle.radius {}".format(Particle.radius))
    temp_range = np.linspace(0.5,1.45,30)
    mean_temps = np.zeros(len(temp_range))
    number_jumps_per_temp = np.zeros(len(temp_range))
    num_per_temp = 5
    temp_jumps = np.zeros(((num_per_temp+2)*len(temp_range), 2))
    jump_instances = []
    counter = 0
    for temp_index, temperature_desired in enumerate(temp_range):
        for rand_attempt in range(num_per_temp):
            print("Beginning iteration for desired temp = " + str(temperature_desired))
            (paths, simulation, temperatures) = serial_simulation(1, label_particles=True, nucleation=False, speed_up=25, load_data=False,
                                sim_name='lattice_diffusion_2d', record_potential=False, lattice_structure=True,
                                desired_temp=temperature_desired, temp_change_finish=3000, temp_change_start=1500)
            
            # start analysis
            # for mean temp we want mean temp after step 3000
            temp_jumps[counter][0] = sum(temperatures[3000:])/len(temperatures[3000:])
            # grid locations for imaginary particles
            grid_locations = paths[1][:25][:,0:2]
            num_real_particles = simulation.num_particles - 25
            dist_between_absorption_sites = (paths[1][0][0] - paths[1][1][0])**2 + (paths[1][0][1] - paths[1][1][1])**2

            # 0 is the first, index 5 to the right, 6 diagonal
            dist_between_diagonal_absorption_sites = (paths[1][0][0] - paths[1][6][0])**2 + (paths[1][0][1] - paths[1][6][1])**2

            


            grid_locs_arr = np.zeros((num_real_particles, len(paths[1:])))
            absorption_indexes_arr = np.zeros((num_real_particles, len(paths[1:])))
            atom_indexes_arr = np.zeros((num_real_particles, len(paths[1:])))
            for i in range(1,len(paths[1:])):
                locations = paths[i][25:][:,0:2]
                (closest_grid_locations, indexes_atoms) = find_closest_grid_locs(locations, grid_locations, num_real_particles)
                
                grid_locs_arr[:,i] = closest_grid_locations
                atom_indexes_arr[:,i] = indexes_atoms
            imaginary_grid_locations = paths[1][:25][:,0:2]


            num_sites_occupied_before_jump_list = []
            percentage_available_sites_occupied_list = []
            num_sites_occupied_before_jump_list_diagonal = []
            percentage_available_sites_occupied_list_diagonal = []

            count_num_jumps = 0
            for i in range(2,len(paths[2:])):
                # this implies a particle transitioned
                if not (grid_locs_arr[:,i] == grid_locs_arr[:,i-1]).all():
                    # now find the particle that moved
                    for j in range(num_real_particles):
                        if grid_locs_arr[j,i] != grid_locs_arr[j,i-1]:
                            count_num_jumps += 1
                            #index here is the index of the closest absorption site
                            print("index_absorption: " +str(grid_locs_arr[j,i-1]))
                            
                            #index here is the index of the atom
                            print("index_atom: " +str(atom_indexes_arr[j,i]))
                            
                            #closest absorption site before jump
                            grid_locs_arr[j,i-1]
                            #now need to calculate the number of particles which has the closest absorption site
                            #which is within 2r of this site
                            
                            #1. find list of absorption sites within 2r before jump (this doesnt change with i)
                            sites, sites_diagonal = find_indexes_of_sites_close(int(grid_locs_arr[j,i-1]), imaginary_grid_locations, dist_between_absorption_sites
                                                    , dist_between_diagonal_absorption_sites)
                            
                            #2. check how many of these sites were occupied for i-1
                            count_sites = len(sites)
                            count_sites_diagonal = len(sites_diagonal)
                            count_sites_total = count_sites + count_sites_diagonal
                            occupied = 0
                            for abs_site_index in sites:
                                for closest_site_index in grid_locs_arr[:,i-1]:
                                    if int(closest_site_index) == abs_site_index:
                                        occupied += 1
                            occupied_diagonal = 0
                            for abs_site_index in sites_diagonal:
                                for closest_site_index in grid_locs_arr[:,i-1]:
                                    if int(closest_site_index) == abs_site_index:
                                        occupied_diagonal += 1
                            occupied_total = occupied + occupied_diagonal
                            
                            print("count_sites = " + str(count_sites))
                            print("count_sites_diagonal = " + str(count_sites_diagonal))
                            print("count_sites_total = " + str(count_sites_total))
                            print("count_occupied = " + str(occupied))
                            print("count_occupied_diagonal = " + str(occupied_diagonal))
                            print("count_occupied_total = " + str(occupied_total))
                            occupied_percentage = occupied/count_sites
                            occupied_diagonal_percentage = occupied_diagonal/count_sites_diagonal
                            occupied_total_percentage = occupied_total/count_sites_total
                            jump_instance = [count_sites, count_sites_diagonal, count_sites_total, occupied, 
                                            occupied_diagonal,occupied_total,occupied_percentage,occupied_diagonal_percentage,occupied_total_percentage,
                                            temperatures[i][0]]
                            print(jump_instance)
                            jump_instances.append(np.array(jump_instance))
                            
            temp_jumps[counter][1] = count_num_jumps
            counter += 1
    jump_instances_np = np.array(jump_instances)
    np.save('./data/temp_jumps_6.npy', temp_jumps)
    np.save('./data/jump_instances_6.npy', jump_instances_np)

# d= distance_point_to_wall((0,0),(10,0),10,10)
# print(d)
# global

if __name__ == '__main__':
    main()
