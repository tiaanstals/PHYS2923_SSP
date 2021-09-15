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
    #final force F(r)=-du(r)/dr=24*(epsilon/sigma)*(2*(sigma/r)^13 - (sigma/r)^6) = constant*(repulsive + attractive)
    lf = const*(repulsive + attractive)
    return lf

def generate_square_matrix(n, x_mid, y_mid):
    spacing = 0.01*Particle.radius
    first_x = x_mid - n*Particle.radius
    first_y = y_mid - n*Particle.radius
    x = []
    y = []
    for i in range(n):
        for j in range(n):
            x_coord = first_x + i*2*Particle.radius 
            y_coord = first_y + j*2*Particle.radius
            if isclose(x_mid, x_coord) and isclose(y_mid,y_coord):
                continue
            x.append(x_coord)
            y.append(y_coord)
    return [x,y]