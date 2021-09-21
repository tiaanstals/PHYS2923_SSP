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
    num_particles = 35
    num_steps = 5000
    velocity_scaler = 0.1
    lim = 30
    x_lim, y_lim = lim, lim
    WINDOW_WIDTH = 600
    WINDOW_HEIGHT = 600
    CONTROL_SPACE = 100
    SPACING_TEXT = 15
    DT = 0.0005
    EPSILON=3
    SIGMA=1
    CUTOFF = 2.5*SIGMA
    SIGMA_6 = SIGMA*SIGMA*SIGMA*SIGMA*SIGMA*SIGMA
    LJ_FORCE_SHIFT=-lj_force_cutoff(CUTOFF, SIGMA, EPSILON, SIGMA_6)

    def __init__(self, loaded_params):
        self.num_particles = loaded_params.num_particles
        self.num_steps = loaded_params.num_steps
        self.velocity_scaler = loaded_params.velocity_scaler
        self.lim = loaded_params.lim 
        self.x_lim = loaded_params.x_lim
        self.y_lim = loaded_params.y_lim
        self.WINDOW_WIDTH = loaded_params.WINDOW_WIDTH
        self.WINDOW_HEIGHT = loaded_params.WINDOW_HEIGHT
        self.CONTROL_SPACE = loaded_params.CONTROL_SPACE
        self.SPACING_TEXT = loaded_params.SPACING_TEXT
        self.DT = loaded_params.DT
        self.EPSILON = loaded_params.EPSILON
        self.SIGMA = loaded_params.SIGMA
        self.SIGMA_6 = loaded_params.SIGMA_6
        self.LJ_FORCE_SHIFT = loaded_params.LJ_FORCE_SHIFT
        

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

def draw_line(window, coord_1, coord_2):
    pygame.draw.line(window, BLACK, coord_1, coord_2)
            
def draw_particle(window, rad, x, y, color_inner, color_outer):
            """
            Create a graphical representation of this particle for visualization.
            win is the graphics windown in which the particle should be drawn.
            
            rad is the radius of the particle"""
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
    
    def set_new_acc(self, new_ax, new_ay):
        self.ax += new_ax
        self.ay += new_ay

    def move(self):
        """ if a particle collides, it cannot continue with the last vx/vy components
        Therefore we must update the vx/vy components using the new acceleration, and then move
        This still has some issues, such as a particle might not move enough to get outside the 
        radius of another particle
        https://www.ccp5.ac.uk/sites/www.ccp5.ac.uk/files/Democritus/Theory/verlet.html#velver
        """
        if not self.constant_particle:
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
    
    def compute_wall_forces(self, box):
        energy_before = self.energy
        BOTTOM_LEFT = (0, 0)
        BOTTOM_RIGHT = (box.box_x, 0)
        TOP_LEFT = (0, box.box_y)
        TOP_RIGHT = (box.box_x,box.box_y)
        
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
        # vertical wall
        elif vertical_count > 0:
            self.new_ax = -2*self.vx/Particle.dt
        #horizontal wall
        elif horizontal_count > 0:
            self.new_ay = -2*self.vy/Particle.dt
        
        energy_after = self.energy_plus_dt
        if isclose(energy_before,0):
            return
        ratio = energy_after/energy_before
        # if vertical_count + horizontal_count > 0:
        #     print(ratio)
        #     self.print()
        return

    def compute_lj_forces(self, other_p):
        """
        Calculate the acceleration on each 
        particle as a  result of each other 
        particle. 
        """ 
        # calculate x and y distances between the two points
        # vector points from other_p to current particle
        rx = self.x - other_p.x
        ry = self.y - other_p.y
        r2 = sqrt(rx*rx+ry*ry)

        lj_f = lj_force(r2)
        if isclose(lj_f, 0):
            return
        # old_ax = self.new_ax
        energy_before = self.energy + other_p.energy
        # if lj_f is positive, acceleration should be in the direction of the vector which points from p2 to p1 (repulsive)
        # if lj_f is negative, acceleration should be in the direction of the vector which points from p1 to p2 (attractive)
        self.set_new_acc(lj_f*rx/r2,lj_f*ry/r2)
        # by newtons third law
        other_p.set_new_acc(-lj_f*rx/r2,-lj_f*ry/r2)
        return
    
    def change_temp(self, change):
        energy_before = self.energy
        self.vx = self.vx + self.vx*change
        self.vy = self.vx + self.vx*change
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

    
###########################
# MATH FUNCTIONS #
###########################
def distance_point_to_wall(WALL_START, WALL_END, x, y):
    wall_start_x, wall_start_y = WALL_START
    wall_end_x, wall_end_y = WALL_END
     
    # avoid square root on the bottom (expensive operation) by squaring the top and squaring in closer_than_radius_function
    d = abs((wall_end_x - wall_start_x)*(wall_start_y-y)-(wall_start_x-x)*(wall_end_y-wall_start_y))**2/((wall_end_x-wall_start_x)**2 + (wall_end_y-wall_start_y)**2)
    return d

radius_dist = (Particle.radius)
def closer_than_radius(distance):
    if distance <= radius_dist:
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
    lf = const*(repulsive + attractive) + Simulation.LJ_FORCE_SHIFT
    return lf

def generate_square_matrix(n, x_mid, y_mid,nucleation):
    cube_2 = 1.2*cube_root(2)
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

##################
# Initialization #
##################
def make_particles(n, temp_scale, nucleation, locations):
    """Construct a list of n particles in two dimensions, initially distributed
    evenly but with random velocities. The resulting list is not spatially
    sorted."""
    seed(2000)
    particles = [Particle(0, 0, 0, 0, 0, 0, _) for _ in range(n)]
    next_p = len(particles)
    # Make sure particles are not spatially sorted
    shuffle(particles)
    vx_list = []
    vy_list = []
    if locations:
        for i, p in enumerate(particles):
            p.x = Simulation.x_lim/2 + locations[0][i]
            p.y = Simulation.y_lim/2 + locations[1][i]
            p.xy = 0
            p.vy = 0
            p.ax = 0
            p.ay = 0
            vx_list.append(p.vx)
            vy_list.append(p.vy)
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
    if nucleation:
        new_particle = Particle(Simulation.x_lim/2, Simulation.y_lim/2,0,0,0,0,next_p)
        new_particle.constant_particle = False
        particles.append(new_particle)
    # make sure no particles are overlapping
    overlap = True
    if locations:
        overlap = False
    while overlap:
        collision_detected = False
        for i, p in enumerate(particles):
            for p2 in particles[i+1:]:
                if round(sqrt((p.x-p2.x)**2 + (p.y-p2.y)**2),2) < (1.5*Particle.radius):
                    # print("Particle 1: [" + str(p.x) + "," + str(p.y) + "]")
                    # print("Particle 2: [" + str(p2.x) + "," + str(p2.y) + "]")
                    # print("(px - p2x)**2 = " + str((p.x-p2.x)**2))
                    # print("(py - p2y)**2 = " + str((p.y-p2.y)**2))
                    if p.constant_particle:
                        p2.x = uniform(0+Particle.radius, Simulation.x_lim-Particle.radius)
                        p2.y = uniform(0+Particle.radius, Simulation.x_lim-Particle.radius)
                    else:
                        p.x = uniform(0+Particle.radius, Simulation.x_lim-Particle.radius)
                        p.y = uniform(0+Particle.radius, Simulation.x_lim-Particle.radius)
                    collision_detected = True
        if not collision_detected:
            overlap = False
                
    return particles



#####################
# Serial Simulation #
#####################
def serial_simulation(update_interval=1, label_particles=False, normalize_energy=True, nucleation=False, speed_up=5, squeeze_box=False, load_data=False,
                        sim_name='latest'):

    # Create particles
    Simulation = Simulation()
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
        # x = np.multiply(x,cube_2)
        # y = np.multiply(y,cube_2)
        # locations = [x, y]
        # Simulation.num_particles = len(x)

        # square
        # x = [Particle.radius, -Particle.radius, 0,0, Particle.radius, -Particle.radius]
        # y = [0,0,Particle.radius,-Particle.radius,Particle.radius, -Particle.radius]
        # x = np.multiply(x,cube_2)
        # y = np.multiply(y,cube_2)
        # locations = [x, y]
        # locations = [x, y]
        # Simulation.num_particles = len(x)

        # locations = generate_square_matrix(3,0,0, nucleation)
        # Simulation.num_particles = len(locations[0])
        locations = None
        particles = make_particles(Simulation.num_particles, Simulation.velocity_scaler, nucleation, locations)
        initial_energy = reduce(lambda x, p: x + p.energy, particles, 0)
        
        clock = pygame.time.Clock()
        myfont = pygame.font.Font('Roboto-Medium.ttf', 10)
        # https://pygamewidgets.readthedocs.io/
        # xcoord, ycoord, width, height, min, max
        # Perform simulation
        start = time()
        running = True
        paths = np.zeros((Simulation.num_steps, len(particles), 4))
        colours = np.zeros(len(particles))
        # squeez box
        # by 50% of the simulation, the box should be x times its original size (x<1)
        # every how many steps should the box decrease by 1 unit?
        # number of steps to accomplish squeeze = Simulation.num_steps/2
        squeeze_size = 0.5
        # amount of change in box limits:
        # final box_x = x*Simulation.x_lim
        final_box_x = squeeze_size*Simulation.x_lim
        # number of increments required:
        # final x_lim - final_box_x
        change_amount = Simulation.x_lim - final_box_x
        print("change amount "+ str(change_amount))
        # every step how much should the box decrease
        step_change_squeeze = change_amount/Simulation.num_steps
        print("step_change_squeeze "+ str(step_change_squeeze))

        box = Box(Simulation.x_lim, Simulation.y_lim)
        box_size = np.zeros((Simulation.num_steps, 1))
        for step in tqdm(range(1,Simulation.num_steps)):
            #compute forces
            if squeeze_box:
                box.set_x(box.box_x - step_change_squeeze)
                box_size[step] = box.box_x
            for i, p in enumerate(particles):
                colours[p.id] = 0 if not p.constant_particle else 1
                paths[step][p.id][0] = p.x
                paths[step][p.id][1] = p.y
                paths[step][p.id][2] = p.energy
                p.new_ax = 0
                p.new_ay = 0

                #compute collision interactions
                
                for p2 in particles:
                    if p2.id == p.id:
                        continue
                    p.compute_lj_forces(p2)
                p.compute_wall_forces(box) 
            
            # Move particles
            for p in particles:
                p.move()
        np.save('./simulations/' + sim_name + '/'+ sim_name + '.npy', paths)
        with open('./simulations/' + sim_name + '/' + 'simulation_params.pkl', 'wb') as output:
            pickle.dump(Simulation, output, pickle.HIGHEST_PROTOCOL)
    else:
        paths = np.load('./simulations/' + sim_name + '/'+ sim_name + '.npy')
        with open('./simulations/' + sim_name + '/' + 'simulation_params.pkl', 'wb') as inp:
            sim_params = pickle.load(inp)
            Simulation = Simulation(sim_params)

    # Initialize visualization
    window = init_graphics()
    slider = Slider(window, 0, Simulation.WINDOW_HEIGHT + 2*Simulation.SPACING_TEXT, Simulation.WINDOW_WIDTH, 10, min=1, max=1000, step=.05, initial=speed_up)
    counter = 0
    speed_up = speed_up
    
    while True:
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
        draw_line(window, (0,Simulation.WINDOW_HEIGHT), (Simulation.WINDOW_WIDTH, Simulation.WINDOW_HEIGHT))
        
        
        input_speed = slider.getValue()
        speed_up = input_speed
        label_speed = myfont.render("Speed: " + str(speed_up), 2, BLACK )
        window.blit(label_speed, (1, Simulation.WINDOW_HEIGHT + Simulation.SPACING_TEXT))
        bottom_left_corner = myfont.render("0,0", 1, BLUE)
        window.blit(bottom_left_corner, (0, 0))
        top_left_corner = myfont.render("0,1", 1, BLUE)
        window.blit(top_left_corner, (0, Simulation.WINDOW_HEIGHT-15))     
        bottom_right_corner = myfont.render("1,0", 1, BLUE)
        window.blit(bottom_right_corner, (Simulation.WINDOW_WIDTH-15, 0))    
        top_right_corner = myfont.render("1,1", 1, BLUE)
        window.blit(top_right_corner, (Simulation.WINDOW_WIDTH-15, Simulation.WINDOW_HEIGHT-15))
        timestep = paths[counter]
        energy_step = np.zeros(len(timestep))

        for i, p in enumerate(timestep):
            x = p[0]
            y = p[1]
            energy_step[i] = p[2]
            # print("[" + str(x) + "," + str(y) + "]")
            color = BLUE if not colours[i] else RED
            draw_particle(window, Particle.radius, x, y, color, LIGHT_BLUE)
            if label_particles:
                label = myfont.render(str(i), 2, BLACK )
                window.blit(label, (to_display_scale(x), to_display_scale(y)))
        mean_e = sum(energy_step)/len(energy_step)
        label = myfont.render("Progress {:2.2%}".format(counter/Simulation.num_steps), 2, BLACK )
        window.blit(label, (1, Simulation.WINDOW_HEIGHT + 1))
        if squeeze_box:
            draw_line(window, (to_display_scale(box_size[counter]),Simulation.WINDOW_HEIGHT), (to_display_scale(box_size[counter]), 0))

        label_e = myfont.render("Energy {:2}".format(mean_e), 2, BLACK )
        window.blit(label_e, (1, Simulation.WINDOW_HEIGHT + 3*Simulation.SPACING_TEXT))
        pygame_widgets.update(event)
        pygame.display.update()
        clock.tick(30)
        counter += int(speed_up)
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
    max_particle_speed = Particle.dt * Particle.radius
    print("max_particle_speed" + str(max_particle_speed))
    serial_simulation(1, True, nucleation=True, speed_up=25, load_data='latest')
    

# d= distance_point_to_wall((0,0),(10,0),10,10)
# print(d)
# global

if __name__ == '__main__':
    main()