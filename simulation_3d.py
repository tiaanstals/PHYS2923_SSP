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
import vpython
from vpython import vector, curve, sphere, color, canvas, points


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
    num_particles = 15
    num_steps = 10000
    velocity_scaler = 0.9
    lim = 20
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

class Box(object):
    def __init__(self, box_x, box_y):
        self.box_x = box_x
        self.box_y = box_y
    
    def set_x(self,new_box_x):
        self.box_x = new_box_x

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

def init_graphics3d():
    win = Simulation.WINDOW_HEIGHT
    L = Simulation.WINDOW_HEIGHT # container is a cube L on a side
    gray = color.gray(0.7) # color of edges of container

    animation = canvas( width=win, height=win, align='left', center=vector(L/2,L/2,L/2))
    animation.range = L
    animation.title = 'A "hard-sphere" gas'
    s = """  Theoretical and averaged speed distributions (meters/sec).
    Initially all atoms have the same speed, but collisions
    change the speeds of the colliding atoms. One of the atoms is
    marked and leaves a trail so you can follow its path.
    
    """
    animation.caption = s

    d = L
    r = Simulation.WINDOW_HEIGHT/150
    # boxbottom = curve(color=gray, radius=r)
    # boxbottom.append([vector(0,0,0), vector(d,0,0), vector(d,d,0), vector(0,d,0), vector(0,0,0)])
    # boxtop = curve(color=gray, radius=r)
    # boxtop.append([vector(0,0,d), vector(d,0,d), vector(d,d,d), vector(0,d,d), vector(0,0,d)])
    # vert1 = curve(color=gray, radius=r)
    # vert2 = curve(color=gray, radius=r)
    # vert3 = curve(color=gray, radius=r)
    # vert4 = curve(color=gray, radius=r)
    # vert1.append([vector(0,0,0), vector(0,0,d)])
    # vert2.append([vector(d,0,0), vector(d,0,d)])
    # vert3.append([vector(d,d,0), vector(d,d,d)])
    # vert4.append([vector(0,d,0), vector(0,d,d)])

    
    return animation

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

    display_radius = 0.8*to_display_scale(radius)

    dt = Simulation.DT

    energy_correction = 1
    epsilon=Simulation.EPSILON
    sigma=Simulation.SIGMA


    def __init__(self, x, y, z, vx, vy, vz, ax, ay, az, id):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.ax = ax
        self.ay = ay
        self.az = az
        self.new_ax = 0
        self.new_ay = 0
        self.id = id
        self.collision = False
        self.constant_particle = False
        self.potential_energy = 0
        self.force_on_wall = 0
        self.lattice_position = False
        self.dont_move_overlap = False
    
    def set_new_acc(self, new_ax, new_ay, new_az):
        self.ax += new_ax
        self.ay += new_ay
        self.az += new_az

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
            vz_half = self.vz + (self.az*self.dt)/2

            self.x = self.x + vx_half * self.dt
            self.y = self.y + vy_half * self.dt 
            self.z = self.z + vz_half * self.dt 

            self.vx = vx_half + (self.dt*self.new_ax)/2
            self.vy = vy_half + (self.dt*self.new_ay)/2
            self.vz = vz_half + (self.dt*self.new_az)/2
            self.ax = self.new_ax
            self.ay = self.new_ay
            self.az = self.new_az

    
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
    
    def compute_wall_forces(self, box):
        limit = box.box_x
        if (self.x > limit and self.vx > 0) or (self.x < 0 and self.vx < 0):
            self.new_ax = -2*self.vx/Particle.dt
            self.force_on_wall += abs(2*self.vx/Particle.dt)
        elif (self.y > limit and self.vy > 0) or (self.y < 0 and self.vy < 0):
            self.new_ay = -2*self.vy/Particle.dt
            self.force_on_wall += abs(2*self.vy/Particle.dt)
        elif (self.z > limit and self.vz > 0) or (self.z < 0 and self.vz < 0):
            self.new_az = -2*self.vz/Particle.dt
            self.force_on_wall += abs(2*self.vz/Particle.dt)
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
        rz = self.z - other_p.z
        r2 = sqrt(rx*rx+ry*ry+rz*rz)

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
        self.set_new_acc(lj_f*rx/r2,lj_f*ry/r2,lj_f*rz/r2)
        # by newtons third law
        other_p.set_new_acc(-lj_f*rx/r2,-lj_f*ry/r2,-lj_f*rz/r2)
        return r2
    
    def compute_lattice_forces(self, other_p):
        rx = self.x - other_p.x
        ry = self.y - other_p.y
        rz = self.z - other_p.z
        r2 = sqrt(rx*rx+ry*ry+rz*rz)

        lj_f = lj_force_attractive_only(r2)
        if isclose(lj_f, 0):
            return r2
        # old_ax = self.new_ax

        # if lj_f is positive, acceleration should be in the direction of the vector which points from p2 to p1 (repulsive)
        # if lj_f is negative, acceleration should be in the direction of the vector which points from p1 to p2 (attractive)

        if isclose(r2, 0):
            return 0
        self.set_new_acc(lj_f*rx/r2,lj_f*ry/r2,lj_f*rz/r2)
        # by newtons third law
        other_p.set_new_acc(-lj_f*rx/r2,-lj_f*ry/r2,-lj_f*rz/r2)
        return r2
    
    def change_temp(self, change):
        energy_before = self.energy
        self.vx = self.vx + self.vx*change
        self.vy = self.vx + self.vx*change
        self.vz = self.vz + self.vz*change
        return self.energy - energy_before
    
    @property
    def energy(self):
        """Return the kinetic energy of this particle."""
        return 0.5 * (self.vx ** 2 + self.vy ** 2 + self.vz**2)
    
    @property 
    def energy_plus_dt(self):
        """Return the kinetic energy in 1 step"""
        vx_half = self.vx + (self.ax*self.dt)/2
        vy_half = self.vy + (self.ay*self.dt)/2
        vz_half = self.vz + (self.az*self.dt)/2
        vx = vx_half + (self.dt*self.new_ax)/2
        vy = vy_half + (self.dt*self.new_ay)/2
        vz = vz_half + (self.dt*self.new_az)/2
        return 0.5 * (vx ** 2 + vy ** 2 + vz**2)

    @property 
    def total_velocity(self):
        return sqrt(self.vx**2+self.vy**2 + self.vz**2)

    
###########################
# MATH FUNCTIONS #
###########################
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
    const = 50*24*Simulation.EPSILON/Simulation.SIGMA
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
    attractive = -inv_r_6**Simulation.SIGMA_6
    lu = const*(repulsive + attractive)
    return lu


def generate_square_matrix(n, x_mid, y_mid,nucleation):
    cube_2 = cube_root(2)
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

def generate_square_matrix_3d(n, x_mid, y_mid, z_mid, nucleation):
    cube_2 = cube_root(2)
    first_x = x_mid - cube_2*n*Particle.radius
    first_y = y_mid - cube_2*n*Particle.radius
    first_z = z_mid - cube_2*n*Particle.radius
    x = []
    y = []
    z = []
    for i in range(2*n+1):
        for j in range(2*n+1):
            for k in range(2*n+1):
                x_coord = first_x + i*cube_2*Particle.radius 
                y_coord = first_y + j*cube_2*Particle.radius
                z_coord = first_z + k*cube_2*Particle.radius
                if isclose(x_mid, x_coord) and isclose(y_mid,y_coord) and nucleation and isclose(z_mid,z_coord):
                    continue
                x.append(x_coord)
                y.append(y_coord)
                z.append(z_coord)
    x_mid = 4*cube_root(2)
    y_mid = 0
    z_mid = 0
    first_x = x_mid - cube_2*n*Particle.radius
    first_y = y_mid - cube_2*n*Particle.radius
    first_z = z_mid - cube_2*n*Particle.radius
    for i in range(2*n+1):
        for j in range(2*n+1):
            for k in range(2*n+1):
                x_coord = first_x + i*cube_2*Particle.radius 
                y_coord = first_y + j*cube_2*Particle.radius
                z_coord = first_z + k*cube_2*Particle.radius
                if isclose(x_mid, x_coord) and isclose(y_mid,y_coord) and nucleation and isclose(z_mid,z_coord):
                    continue
                x.append(x_coord)
                y.append(y_coord)
                z.append(z_coord)
    x_mid = -4*cube_root(2)
    y_mid = 0
    z_mid = 0
    first_x = x_mid - cube_2*n*Particle.radius
    first_y = y_mid - cube_2*n*Particle.radius
    first_z = z_mid - cube_2*n*Particle.radius
    for i in range(2*n+1):
        for j in range(2*n+1):
            for k in range(2*n+1):
                x_coord = first_x + i*cube_2*Particle.radius 
                y_coord = first_y + j*cube_2*Particle.radius
                z_coord = first_z + k*cube_2*Particle.radius
                if isclose(x_mid, x_coord) and isclose(y_mid,y_coord) and nucleation and isclose(z_mid,z_coord):
                    continue
                x.append(x_coord)
                y.append(y_coord)
                z.append(z_coord)
    return [x,y,z]

def generate_square_matrix_2halfd(n, x_mid, y_mid, z_mid, nucleation):
    cube_2 = cube_root(2)
    first_x = x_mid - cube_2*n*Particle.radius
    first_y = y_mid - cube_2*n*Particle.radius
    first_z = z_mid - cube_2*n*Particle.radius
    x = []
    y = []
    z = []
    for i in range(2*n+1):
        for j in range(2*n+1):
            x_coord = first_x + i*cube_2*Particle.radius 
            y_coord = first_y + j*cube_2*Particle.radius
            z_coord = 0 + cube_2*Particle.radius
            if isclose(x_mid, x_coord) and isclose(y_mid,y_coord) and nucleation and isclose(z_mid,z_coord):
                continue
            x.append(x_coord)
            y.append(y_coord)
            z.append(z_coord)
    return [x,y,z]

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
def make_particles(n, temp_scale, nucleation, locations, fast_particle=False, lattice_structure=False, lattice_size=1):
    """Construct a list of n particles in two dimensions, initially distributed
    evenly but with random velocities. The resulting list is not spatially
    sorted."""
    seed(2000)
    particles = [Particle(0, 0, 0, 0, 0, 0, 0, 0, 0, _) for _ in range(n)]
    next_p = len(particles)
    # Make sure particles are not spatially sorted
    shuffle(particles)
    vx_list = []
    vy_list = []
    vz_list = []
    if locations is not None:
        for i, p in enumerate(particles):
            p.x = Simulation.x_lim/2 + locations[0][i]
            p.y = Simulation.y_lim/2 + locations[1][i]
            p.z = Simulation.y_lim/2 + locations[2][i]
            p.xy = 0
            p.vy = 0
            p.vz = 0
            p.ax = 0
            p.ay = 0
            p.az = 0
            vx_list.append(p.vx)
            vy_list.append(p.vy)
            vz_list.append(p.vz)
        # for i in range(round(len(locations[0])/2)):
        #     particle = Particle(0, 0, 0, 0, 0, 0, 0, 0, 0,next_p)
        #     particle.x = uniform(0+Particle.radius, Simulation.x_lim-Particle.radius)
        #     particle.y = uniform(0+Particle.radius, Simulation.y_lim-Particle.radius)
        #     particle.z = uniform(0+Particle.radius, Simulation.y_lim-Particle.radius)
        #     particle.vx = temp_scale*uniform(-Simulation.x_lim, Simulation.x_lim)
        #     particle.vy = temp_scale*uniform(-Simulation.y_lim, Simulation.y_lim)
        #     particle.vz = temp_scale*uniform(-Simulation.y_lim, Simulation.y_lim)
        #     particles.append(particle)
        #     vx_list.append(particle.vx)
        #     vy_list.append(particle.vy)
        #     vz_list.append(particle.vz)
        #     next_p += 1                
        #     print(next_p)
    elif lattice_structure:
        # generates a 2d array with coordinates for the lattice
        locations = generate_square_matrix_2halfd(lattice_size,0,0,0, nucleation=False)
        # locations = rotate_square_matrix(locations, 20)
        # first create lattice position particles
        particles = []
        for i in range(len(locations[0])):
            particles.append(Particle(0, 0, 0, 0, 0, 0, 0, 0, 0,i))
            # this property means the "particle" is actually just a preferred position
            particles[i].lattice_position = True
            particles[i].x = Simulation.y_lim/2 + locations[0][i]
            particles[i].y = Simulation.y_lim/2 + locations[1][i]
            particles[i].z = locations[2][i]
        print(len(particles))
        next_p = i + 1
        minimum = min(locations[0])
        maximum = max(locations[0])
        print(minimum)
        print(maximum)
        for i in range(len(locations[0])):
            populate = random() < 0.9
            particle = Particle(0, 0, 0, 0, 0, 0, 0, 0, 0,next_p)
            particle.x = Simulation.y_lim/2 + locations[0][i]
            particle.y = Simulation.y_lim/2 +  locations[1][i]
            particle.z = locations[2][i]
            particle.dont_move_overlap = True
            print(populate)
            if populate:
                particles.append(particle)
                vx_list.append(particle.vx)
                vy_list.append(particle.vy)
                vz_list.append(particle.vz)
                next_p += 1
            print(next_p)
        
        # add in some other particles
        for i in range(round(len(locations[0])/2)):
            particle = Particle(0, 0, 0, 0, 0, 0, 0, 0, 0,next_p)
            particle.x = uniform(0+Particle.radius, Simulation.x_lim-Particle.radius)
            particle.y = uniform(0+Particle.radius, Simulation.y_lim-Particle.radius)
            particle.z = uniform(0+Particle.radius, Simulation.y_lim-Particle.radius)
            particle.vx = temp_scale*uniform(-Simulation.x_lim, Simulation.x_lim)
            particle.vy = temp_scale*uniform(-Simulation.y_lim, Simulation.y_lim)
            particle.vz = temp_scale*uniform(-Simulation.y_lim, Simulation.y_lim)
            particles.append(particle)
            vx_list.append(particle.vx)
            vy_list.append(particle.vy)
            vz_list.append(particle.vz)
            next_p += 1                
            print(next_p)
    else:
        for p in particles:
            # Distribute particles randomly in our box
            p.x = uniform(0+Particle.radius, Simulation.x_lim-Particle.radius)
            p.y = uniform(0+Particle.radius, Simulation.y_lim-Particle.radius)
            p.z = uniform(0+Particle.radius, Simulation.y_lim-Particle.radius)
            # Assign random velocities within a bound
            p.vx = temp_scale*uniform(-Simulation.x_lim, Simulation.x_lim)
            p.vy = temp_scale*uniform(-Simulation.y_lim, Simulation.y_lim)
            p.vz = temp_scale*uniform(-Simulation.y_lim, Simulation.y_lim)
            vx_list.append(p.vx)
            vy_list.append(p.vy)
            vz_list.append(p.vz)
            p.ax = 0
            p.ay = 0
            p.az = 0
    # let total linear momentum be initially 0
    mean_vx = sum(vx_list)/len(vx_list)
    mean_vy = sum(vy_list)/len(vy_list)
    mean_vz = sum(vz_list)/len(vz_list)
    for p in particles:
        p.vx -= mean_vx
        p.vy -= mean_vy
        p.vz -= mean_vz
    if nucleation:
        new_particle = Particle(Simulation.x_lim/2, Simulation.y_lim/2,Simulation.y_lim/2,0,0,0,0,0,0,next_p)
        new_particle.potential_energy = 0
        new_particle.constant_particle = False
        particles.append(new_particle)
        next_p += 1
    if fast_particle:
        speed = 30
        acc = 20
        new_particle = Particle(2*Particle.radius,Simulation.y_lim/2,speed,0,acc,0,next_p)
        particles.append(new_particle)
        new_particle_2 = Particle(2*Particle.radius,Simulation.y_lim/2 + Particle.radius,speed,0,acc,0,next_p+1)
        particles.append(new_particle_2)
        new_particle_3 = Particle(2*Particle.radius,Simulation.y_lim/2 - Particle.radius,speed,0,acc,0,next_p+2)
        particles.append(new_particle_3)
        new_particle_4 = Particle(Simulation.x_lim - 2*Particle.radius,Simulation.y_lim/2,-speed,0,-acc,0,next_p+3)
        particles.append(new_particle_4)
        new_particle_5 = Particle(Simulation.x_lim - 2*Particle.radius,Simulation.y_lim/2 + Particle.radius,-speed,0,-acc,0,next_p+4)
        particles.append(new_particle_5)
        new_particle_6 = Particle(Simulation.x_lim - 2*Particle.radius,Simulation.y_lim/2 - Particle.radius,-speed,0,-acc,0,next_p+5)
        particles.append(new_particle_6)
    # make sure no particles are overlapping
    overlap = True
    if (locations is not None) and (lattice_structure == False):
        overlap = False
    while overlap:
        collision_detected = False
        print(overlap)
        for i, p in enumerate(particles):
            for p2 in particles[i+1:]:
                if round(sqrt((p.x-p2.x)**2 + (p.y-p2.y)**2 + (p.z-p2.z)**2),2) < (1.5*Particle.radius):
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
                        p2.z = uniform(0+2*Particle.radius, Simulation.x_lim-2*Particle.radius)
                        collision_detected = True
                    else:
                        p.x = uniform(0+2*Particle.radius, Simulation.x_lim-2*Particle.radius)
                        p.y = uniform(0+2*Particle.radius, Simulation.x_lim-2*Particle.radius)
                        p.z = uniform(0+2*Particle.radius, Simulation.x_lim-2*Particle.radius)
                        collision_detected = True
                    
        if not collision_detected:
            overlap = False
                
    return particles



#####################
# Serial Simulation #
#####################
def serial_simulation(update_interval=1, label_particles=False, normalize_energy=True, nucleation=False, speed_up=5, squeeze_box=False, load_data=False,
                        sim_name='latest', fast_particle=False, record_potential=False):

    # Create particles
    if not load_data:
        cube_2 = cube_root(2)
        #hexagon
        # equilibrium distance is cuberoot(2)
        # https://www.wolframalpha.com/input/?i=roots+24*%282%281%2Fx%29%5E13-0.5*%281%2Fx%29%5E7%29
        # thus all distances should be cuberoot(3)*value
        # https://www.wolframalpha.com/input/?i=roots+24*%282%281%2Fx%29%5E13-0.5*%281%2Fx%29%5E7%29
        # 
        # x = [Particle.radius/2, Particle.radius, Particle.radius/2, -Particle.radius/2, -Particle.radius,-Particle.radius/2]
        # y = [sqrt(3)*Particle.radius/2, 0, -sqrt(3)*Particle.radius/2,-sqrt(3)*Particle.radius/2, 0, sqrt(3)*Particle.radius/2]
        # z = [0,0,0,0,0,0]
        # x = np.multiply(x,cube_2)
        # y = np.multiply(y,cube_2)
        # locations = [x, y, z]
        # Simulation.num_particles = len(x)

        # conway hexagon
        # r_1 = cube_2
        # r_2 = cube_2*2
        # x = [r_1 * cos(0), r_1 * cos(pi/3), r_1 * cos(2*pi/3), r_1 * cos(pi), r_1 * cos(4*pi/3), r_1 * cos(5*pi/3)]
        # y = [r_1 * sin(0), r_1 * sin(pi/3), r_1 * sin(2*pi/3), r_1 * sin(pi), r_1 * sin(4*pi/3), r_1 * sin(5*pi/3)]
        # x_2 = [r_2 * cos(pi), r_2 * cos(5*pi/6), r_2 * cos(7*pi/6), r_2 * cos(9*pi/6), r_2 * cos(11*pi/6), r_2 * cos(13*pi/6)]
        # y_2 = [r_2 * sin(pi/2), r_2 * sin(5*pi/6), r_2 * sin(7*pi/6), r_2 * sin(9*pi/6), r_2 * sin(11*pi/6), r_2 * sin(13*pi/6)]
        # z = [0,0,0,0,0,0]
        # z2 = [0,0,0,0,0,0]
        # locations = [x+x_2, y+y_2, z+z2]
        # Simulation.num_particles = len(x)

        # square
        # x = [Particle.radius, -Particle.radius, 0,0, Particle.radius, -Particle.radius]
        # y = [0,0,Particle.radius,-Particle.radius,Particle.radius, -Particle.radius]
        # x = np.multiply(x,cube_2)
        # y = np.multiply(y,cube_2)
        # locations = [x, y]
        # locations = [x, y]
        # Simulation.num_particles = len(x)

        locations = generate_square_matrix_3d(1,0,0,0, nucleation)
        # # locations = rotate_square_matrix(locations, 20)
        Simulation.num_particles = len(locations[0])

        # lattice structure
        # lattice_structure=True
        # locations = None


        # locations = None
        particles = make_particles(Simulation.num_particles, Simulation.velocity_scaler, nucleation, locations, fast_particle, lattice_structure=False, lattice_size=3)
        Simulation.num_particles = len(particles)
        print(Simulation.num_particles)
        # Initialize visualization

        

        # Perform simulation
        start = time()
        running = True
        paths = np.zeros((Simulation.num_steps, len(particles), 7))
        rdf_data = np.zeros((Simulation.num_steps, len(particles), len(particles)))
        colours = np.zeros(len(particles))
        lattice_data = np.zeros(len(particles))
        box = Box(Simulation.x_lim, Simulation.y_lim)
        box_size = np.zeros((Simulation.num_steps, 1))
        for p in particles:
            if p.dont_move_overlap:
                colours[p.id] = 1
            else:
                colours[p.id] = 0
            if p.lattice_position:
                lattice_data[p.id] = 1
            else:
                lattice_data[p.id] = 0
            
        for step in tqdm(range(1,Simulation.num_steps)):
            #compute forces
            if squeeze_box:
                box_size[step] = box.box_x
            for i, p in enumerate(particles):
                
                paths[step][p.id][0] = p.x
                paths[step][p.id][1] = p.y
                paths[step][p.id][2] = p.z
                # kinetic energy
                paths[step][p.id][3] = p.energy


                # potential energy
                if record_potential:
                    paths[step][p.id][4] = p.potential_energy

                # forces on walls (used for pressure calcs)
                paths[step][p.id][5] = p.force_on_wall

                # total velocity (since m = 1, p = v)
                paths[step][p.id][6] = p.total_velocity


                p.new_ax = 0
                p.new_ay = 0
                p.new_az = 0
                p.force_on_wall = 0
                p.potential_energy = 0

                #compute collision interactions
                
                for p2 in particles:
                    if p2.id == p.id:
                        continue
                    r = p.compute_lj_forces(p2, record_potential)
                    # this means one particle is a lattice position
                    if r == -1:
                        r = p.compute_lattice_forces(p2)
                    rdf_data[step][p.id][p2.id] = r
                    rdf_data[step][p2.id][p.id] = r
                p.compute_wall_forces(box) 
            
            # Move particles
            for p in particles:
                p.move()
        try:
            os.mkdir('./simulations/' + sim_name)
        except OSError:
            print ("Creation of the directory failed")
        else:
            print ("Successfully created the directory")
        
        np.save('./simulations/' + sim_name + '/'+ sim_name + '.npy', paths)
        np.save('./simulations/' + sim_name + '/'+ sim_name + '_colours.npy', colours)
        np.save('./simulations/' + sim_name + '/'+ sim_name + '_lattice_data.npy', lattice_data)
        np.save('./simulations/' + sim_name + '/'+ sim_name + '_rdf_data.npy', rdf_data)
        with open('./simulations/' + sim_name + '/' + 'simulation_params.pkl', 'wb') as output:
            simulation = Simulation()
            pickle.dump(simulation, output, pickle.HIGHEST_PROTOCOL)
        params_for_analysis = {'x_lim': Simulation.x_lim, 'y_lim': Simulation.y_lim}
        with open('./simulations/' + sim_name + '/' + 'simulation_params_for_analysis.pkl', 'wb') as output:
            pickle.dump(params_for_analysis, output, pickle.HIGHEST_PROTOCOL)
    else:
        paths = np.load('./simulations/' + sim_name + '/'+ sim_name + '.npy')
        colours = np.load('./simulations/' + sim_name + '/'+ sim_name + '_colours.npy')
        lattice_data = np.load('./simulations/' + sim_name + '/'+ sim_name + '_lattice_data.npy')
        with open('./simulations/' + sim_name + '/' + 'simulation_params.pkl', 'rb') as inp:
            sim_params = pickle.load(inp)
            set_sim_params(sim_params, Simulation)
    
    window = init_graphics3d()
    # clock = pygame.time.Clock()
    # myfont = pygame.font.Font('Roboto-Medium.ttf', 10)
    # https://pygamewidgets.readthedocs.io/
    # xcoord, ycoord, width, height, min, max
    counter = 0
    speed_up = speed_up
    atoms = []
    print(lattice_data)
    Simulation.num_particles = len(lattice_data)

    counter = 0
    for i in range(Simulation.num_particles):
        
        x = paths[0][i][0]
        y = paths[0][i][1]
        z = paths[0][i][2]
        green = vpython.color.green
        blue = vpython.color.blue
        if colours[i] == 1:
            color = blue
        else:
            color = green
        if lattice_data[i] != 1:
                # lattice particle
            atoms.append(sphere(pos=vector(x,y,z), radius=Particle.display_radius,color=color))
    
    # num_points = sum(i < 0 for i in lattice_data)
    # points_list = []
    # for i in range(Simulation.num_particles):
    #     x = to_display_scale(paths[0][i][0])
    #     y = to_display_scale(paths[0][i][1])
    #     z = to_display_scale(paths[0][i][2])
    #     if lattice_data[1] == 1:
    #         points_list.append(vector(x,y,z))
    # if points_list:
    #     points(pos=points_list, color=vpython.color.red)

    while True:
        vpython.rate(300)
        timestep = paths[counter]
        energy_step = np.zeros(len(timestep))
        counter_particles = 0
        for i, p in enumerate(timestep):
            # just a point
            if lattice_data[i] == 1:
                # lattice particle
                continue
            x = to_display_scale(p[0])
            y = to_display_scale(p[1])
            z = to_display_scale(p[2])
            # energy_step[i] = p[3]
            # # print("[" + str(x) + "," + str(y) + "]")
            # color = BLUE if not colours[i] else RED
            # lattice_particle = True if lattice_data[i] else False
            atoms[counter_particles].pos = vector(x,y,z)
            counter_particles += 1

        mean_e = sum(energy_step)/len(energy_step)

        counter += 1
        if counter >= Simulation.num_steps:
            counter = 0


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

def main():
    pygame.init()
    print("x_lim {}".format(Simulation.x_lim))
    print("y_lim {}".format(Simulation.y_lim))
    print("Particle.radius {}".format(Particle.radius))
    serial_simulation(1, label_particles=False, nucleation=False, speed_up=25, load_data=False ,sim_name='3d_cube', record_potential=False)
    

# d= distance_point_to_wall((0,0),(10,0),10,10)
# print(d)
# global

if __name__ == '__main__':
    main()