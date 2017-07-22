"""Mountain-car problem simulation library for the reinforcement learning
miniproject.
"""

import pylab as plb
import numpy as np

class MountainCar():
    """A mountain-car problem.

    For the miniproject, you are not meant to change the default parameters
    (mass of the car, etc.)

    Usage: 
        >>> mc = MountainCar()
        
        Set the agent to apply a rightward force (positive in x)
        >>> mc.apply_force(+1) # the actual value doesn't mattter, only the sign
        
        Run an "agent time step" of 1s with 0.01 s integration time step
        >>> mc.simulate_timesteps(n = 100, dt = 0.01)
        
        Check the state variables of the agent, and the reward
        >>> print mc.x, mc.x_d, mc.R

        At some point, one might want to reset the position/speed of the car
        >>> mc.reset()
    """

    def __init__(self, prng, g = 10.0, d = 100.0, H = 10., m = 10.0, 
                force_amplitude = 3.0, reward_amplitude = 1., 
                 reward_threshold = 0.0):
        
        # set internal parameters from constructor call
        self.g = g # gravitational constant
        self.d = d # minima location
        self.H = H # height of the saddle point
        self.m = m # mass of the car
        self.force_amplitude = force_amplitude # amplitude of the force applied by the engine
        self.reward_amplitude = reward_amplitude # value of the reward
        self.reward_threshold = reward_threshold # x-axis threshold for the obtention of reward
        self.prng = prng

        # reset the car variables
        self.reset()

    def reset(self):
        """Reset the mountain car to a random initial position.
        """

        # set position to range [-130; -50]
        self.x = 80 * self.prng.rand() - 130.0 
        #self.x = -60.0
        #print self.x
        
        # set x_dot to range [-5; 5]
        self.x_d = 10.0 * self.prng.rand() - 5.0
        # reset reward
        self.R = 0.0
        # reset time
        self.t = 0.0
        # reset applied force
        self.F = 0.0

    def apply_force(self, direction):
        """Apply a force to the car.

        Only three values of force are possible: 
            right (if direction > 0), 
            left (direction < 0) or
            no force (direction = 0).\
        """
        self.F = np.sign(direction) * self.force_amplitude

    def _h(self, x):
        """Return the value of the landscape function h in x.
        """
        return (x - self.d)**2 * (x + self.d)**2 / ((self.d**4/self.H)+x**2)
        
    def _h_prime(self, x):
        """Return the value of the first derivative of the landscape function h in x.
        """
        c = self.d**4/self.H
        return 2 * x * (x**2 - self.d**2) * (2*c + self.d**2  + x**2) / (c+x**2)**2

    def _h_second(self, x):
        """Return the value of the second derivative of the landscape function h in x.
        """
        c = self.d**4/self.H
        return 2 * (
            - 2 * c**2 * (self.d**2 - 3*x**2) 
            + c * (-self.d**4 + 6*self.d**2 * x**2 + 3*x**4)
            + 3 * self.d**4 * x**2
            + x**6
        ) / (c + x**2)**3

    def _energy(self, x, x_d):
        """Return the total energy of the car with variable x and x_d.
        """
        # note that v and x dot are not the same: v includes the y direction!
        return self.m * (self.g * self._h(x) + 0.5 * (1 + self._h_prime(x)**2) * x_d**2) 

    def simulate_timesteps(self, n = 1, dt = 0.1):
        """Simulate the car dynamics for n timesteps of length dt.
        """

        for i in range(n):
            self._simulate_single_timestep(dt)
        
        self.t += n*dt

        # check for rewards
        self.R = self._get_reward()

    def _simulate_single_timestep(self, dt):
        """Simulate the car dynamics for a single timestep.
        """

        # calculate the second derivative of x (horiz. acceleration)
        alpha = np.arctan(self._h_prime(self.x))
        x_dd = np.cos(alpha) * (self.F / self.m - np.sin(alpha) * (self.g + self._h_second(self.x) * self.x_d**2))

        # update the position and velocity on the x axis
        self.x += self.x_d * dt + 0.5 * x_dd * dt**2
        self.x_d += x_dd * dt

    def _get_reward(self):
        """Check for and return reward.
        """

        # if there's already a reward, we stick to it
        if self.R > 0.0:
            return self.R
        # have we crossed the threshold?
        if self.x >= self.reward_threshold:
            return self.reward_amplitude

        # else no reward
        return 0.0

class MountainCarViewer():
    """Display the state of a MountainCar instance.
    
    Usage: 
        >>> mc = MountainCar()

        >>> mv = MoutainCarViewer(mc)

        Turn matplotlib's "interactive mode" on and create figure
        >>> plb.ion()
        >>> mv.create_figure(n_steps = 200, max_time = 200)
        
        This forces matplotlib to draw the fig. before the end of execution
        >>> plb.draw()
        
        Simulate the MountainCar, visualizing the state
        >>> for n in range(200):
        >>>     mc.simulate_timesteps(100,0.01)
        >>>     mv.update_figure()
        >>>     plb.draw()
    """
    
    def __init__(self, mountain_car):
        assert isinstance(mountain_car, MountainCar), \
                'Argument to MoutainCarViewer() must be a MountainCar instance'
        self.mountain_car = mountain_car

    def create_figure(self, n_steps, max_time, f = None):
        """Create a figure showing the progression of the car.
        
        Call update_car_state susequently to update this figure.

        Parameters:
        -----------
        n_steps  -- number of times update_car_state will be called.
        max_time -- the time the trial will last (to scale the plots).
        f        -- (optional) figure in which to create the plots.
        """

        if f is None:
            self.f = plb.figure()
        else:
            self.f = f

        # create the to store the arrays
        self.times = np.zeros(n_steps + 1)
        self.positions = np.zeros((n_steps + 1,2))
        self.forces = np.zeros(n_steps + 1)
        self.energies = np.zeros(n_steps + 1)

        # Fill the initial values
        self.i = 0
        self._get_values()

        # create the energy landscape plot
        self.ax_position = plb.subplot(2,1,1)
        self._plot_energy_landscape(self.ax_position)
        self.h_position = self._plot_positions()

        # create the force plot
        self.ax_forces = plb.subplot(2,2,3)
        self.h_forces = self._plot_forces()
        plb.axis(xmin = 0, xmax = max_time, 
                 ymin = -1.1 * self.mountain_car.force_amplitude,
                 ymax = 1.1 * self.mountain_car.force_amplitude)
        
        # create the energy plot
        self.ax_energies = plb.subplot(2,2,4)
        self.h_energies = self._plot_energy()
        plb.axis(xmin = 0, xmax = max_time, 
                 ymin = 0.0, ymax =1000.)

    def update_figure(self):
        """Update the figure.

        Assumes the figure has already been created with create_figure.
        """

        # increment 
        self.i += 1
        assert self.i < len(self.forces), \
                "update_figure was called too many times."

        # get the new values from the car
        self._get_values()

        # update the plots
        self._plot_positions(self.h_position)
        self._plot_forces(self.h_forces)
        self._plot_energy(self.h_energies)

    def _get_values(self):
        """Retrieve the relevant car variables for the figure.
        """
        self.times[self.i] = self.mountain_car.t
        self.positions[self.i,0] = self.mountain_car.x
        self.positions[self.i,1] = self.mountain_car.x_d
        self.forces[self.i] = self.mountain_car.F
        self.energies[self.i] = self.mountain_car._energy(
            self.mountain_car.x, self.mountain_car.x_d)

    def _plot_energy_landscape(self, ax = None):
        """plot the energy landscape for the mountain car in 2D.

        Returns the axes instance created. Use plot_energy_landscape to let 
        the module decide whether you have the right modules for 3D plotting.
        """

        # create coordinates for a grid in the x-x_dot space
        X = np.linspace(-160, 160, 61)
        XD = np.linspace(-20, 20, 51)
        X,XD = np.meshgrid(X , XD)

        # calculate the energy in each point of the grid
        E = self.mountain_car._energy(X, XD)

        # display the energy as an image
        if ax is None:
            f = plb.figure()
            ax = plb.axes()
        
        C = ax.contourf(X,XD, E,100)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$\dot x$')
        cbar = plb.colorbar(C)
        cbar.set_label('$E$')

        return ax

    def _plot_positions(self, handles = None):
        """plot the position and trajectory of the car in state space.
        """

        # choose the color of the point according to the force direction
        color = ['r', 'w', 'g'][1 + int(np.sign(self.mountain_car.F))]

        if handles is None:
            # create the plots
            handles = [] # keep the plot objects in this list
            handles.append(plb.plot(
                np.atleast_1d(self.positions[:self.i+1,0]),
                np.atleast_1d(self.positions[:self.i+1,1]),
                ',k'
            )[0])
            handles.append(plb.plot(
                np.atleast_1d(self.positions[self.i,0]),
                np.atleast_1d(self.positions[self.i,1]),
                'o' + color,
                markeredgecolor = 'none',
                markersize = 9,                
            )[0])
            return tuple(handles)
        else:
            # update the plots
            handles[0].set_xdata(np.atleast_1d(self.positions[:self.i+1,0]))
            handles[0].set_ydata(np.atleast_1d(self.positions[:self.i+1,1]))
            handles[1].set_xdata(np.atleast_1d(self.positions[self.i,0]))
            handles[1].set_ydata(np.atleast_1d(self.positions[self.i,1]))
            handles[1].set_color(color)
            return handles

    def _plot_forces(self, handle = None):
        """plot the force applied by the car vs time.
        """
        # create the plots
        if handle is None:
            handle = plb.plot(
                np.atleast_1d(self.times[:self.i+1]),
                np.atleast_1d(self.forces[:self.i+1]),
                ',k',
            )[0]

            plb.xlabel('$t$')
            plb.ylabel('$F$')
            return handle
        else:
            # update the plot
            handle.set_xdata(np.atleast_1d(self.times[:self.i+1]))
            handle.set_ydata(np.atleast_1d(self.forces[:self.i+1]))
            return handle

    def _plot_energy(self, handle = None):
        """plot the energy of the car vs time.
        """
        # create the plots
        if handle is None:
            handle = plb.plot(
                np.atleast_1d(self.times[:self.i+1]),
                np.atleast_1d(self.energies[:self.i+1]),
                'k',
                linewidth = 0.5
            )[0]

            plb.xlabel('$t$')
            plb.ylabel('$E$')
            return handle
        else:
            # update the plot
            handle.set_xdata(np.atleast_1d(self.times[:self.i+1]))
            handle.set_ydata(np.atleast_1d(self.energies[:self.i+1]))
            return handle
