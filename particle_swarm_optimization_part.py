import numpy as np


# This is a PSO (inertia weight) variation
class Particle:

    # Particle class represents a solution inside a swarm
    def __init__(self, no_dim, x_range, v_range):
        # parameter no_dim is "number of dimension"
        # parameter x_range is "Min and Max range of dimension"
        # parameter v_range is "Min and Max range of velocity"

        # particle position in each dimension
        self.x = np.random.uniform(x_range[0], x_range[1], (no_dim, ))

        # particle velocity in each dimension
        self.v = np.random.uniform(v_range[0], v_range[1], (no_dim, ))      # 163 dimension

        self.pbest = np.inf
        self.pbestpos = np.zeros(no_dim)


# Swarm class represents a pool of solution(particle)
class Swarm:
    def __init__(self, no_particle, no_dim, x_range, v_range, iw_range, c):
        # parameter no_partile is "number of particles (solutions)
        # parameter no_dim is "number of dimensions"
        # parameter x_range is "Min and Max value (range) of dimension"
        # parameter v_range is "Min and Max value (range) of velocity"
        # parameter iw_range is "Min and Max value (range) of inertia weight
        # parameter c[0] is "cognitive parameter and c[1] is "social parameter

        self.p = np.array([Particle(no_dim, x_range, v_range) for i in range(no_particle)])     # array of particle
        self.gbest = np.inf         # global best
        self.gbestpos = np.zeros(no_dim, )
        self.x_range = x_range
        self.v_range = v_range
        self.iw_range = iw_range        # range of inertia weight
        self.c0 = c[0]
        self.c1 = c[1]
        self.no_dim = no_dim

    # start optimization
    def optimize(self, function, X, Y, print_step, iter):
        # optimoze (forward_pass, X, Y, 100, 1000)

        # parameter function is "the function to be optimized"
        # parameter X is "input used in forward pass"
        # parameter Y is :"target used to calculate loss"
        # parameter print_step is "print and pause between two adjacent prints
        # parameter iter is "number of iteration"

        for i in range(iter):           # iter = 1000
            for particle in self.p:
                fitness = function(X, Y, particle.x)        # evaluate the particle fitness

                if fitness < particle.pbest:
                    particle.pbest = fitness                # update the personal fitness
                    particle.pbestpos = particle.x.copy()   # update the personal best position

                if fitness < self.gbest:
                    self.gbest = fitness                    # update the global fitness
                    self.gbestpos = particle.x.copy()       # update the global best position

            for particle in self.p:
                iw = np.random.uniform(self.iw_range[0], self.iw_range[1], 1)         # value of inertia weight

                # calculate particle velocity
            # Particle’s velocity v[] = c0 *v[] + c1*rand()*(pbest[] - present[]) + c2*rand()*(gbest[] - present[])
                particle.v = iw * particle.v + (self.c0 * np.random.uniform(0.0, 1.0, (self.no_dim, )) *
                              (particle.pbestpos - particle.x)) + (self.c1 * np.random.uniform(0.0, 1.0,
                                (self.no_dim, )) * (self.gbestpos - particle.x))

                # update particle position
                # present[] = present[] + v[]
                particle.x = particle.x + particle.v

            if i % print_step == 0:
                print(" For iteration", i + 1, "  loss is# : ", fitness)

        print("\n Global best loss is# :  ", self.gbest)


    def get_best_solution(self):

        return self.gbestpos    # return array of paramaters/weights
