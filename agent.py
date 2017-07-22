import sys
import mountaincar
import numpy as np
import scipy.misc
import pylab as plb

"""
MountainCar with SARSA-Lambda.
Author: Dario Pavllo
See https://github.com/dariopavllo/mountaincar-sarsa-lambda
"""

class LearningAgent:
    
    """A learning agent for the mountain-car task.
    """

    def __init__(self, seed=1):
        self.rnd = np.random.RandomState(seed)
        self.mountain_car = mountaincar.MountainCar(self.rnd)
        
        # Initialize constants
        self.x_min = -150.0
        self.x_max = 30.0
        self.x_n = 20 # Number of subdivisions along the position axis

        self.v_min = -15.0
        self.v_max = 15.0
        self.v_n = 5 # Number of subdivisions along the speed axis
        
        self.x_centers = np.linspace(self.x_min, self.x_max, self.x_n)
        self.v_centers = np.linspace(self.v_min, self.v_max, self.v_n)
        self.x_sigma = self.x_centers[1] - self.x_centers[0]
        self.v_sigma = self.v_centers[1] - self.v_centers[0]

    def visualize_trial(self, n_steps = 200):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """
        
        # prepare for the visualization
        plb.ion()
        plb.pause(0.0001)
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.show()
            
        # make sure the mountain-car is reset
        self.mountain_car.reset()

        for n in range(n_steps):
            print('\rt =', self.mountain_car.t,
            sys.stdout.flush())
            
            inputs = self._compute_inputs(self.mountain_car.x, self.mountain_car.x_d)
            outputs = self._compute_outputs(inputs, self.W)
            action = self._select_action(self._softmax(outputs, self.t))
            self.mountain_car.apply_force(action - 1)
            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            # update the visualization
            mv.update_figure()
            plb.show()
            plb.pause(0.0001)

            # check for rewards
            if self.mountain_car.R > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break

    def learn(self, num_trials=100, eta=0.1, gamma=0.95, decay=0.8, t=1.0, t_decay=0.9, w0=0):
        ''' Train this agent.
        
        Parameters:
        num_trial -- the number of trials to train for
        eta       -- learning rate
        gamma     -- discount factor
        decay     -- eligibility trace decay factor (i.e. lambda)
        t         -- initial softmax temperature parameter
        t_decay   -- exponential decay factor for the temperature parameter
        w0        -- initial weight value
        '''
        
        self.W = np.zeros((self.x_n * self.v_n, 3))
        self.W = self.W + w0
        self.t = t
        
        self.results = []
        for trial in range(num_trials):
            inputs = self._compute_inputs(self.mountain_car.x, self.mountain_car.x_d)
            state = self.mountain_car.x, self.mountain_car.x_d
            outputs = self._compute_outputs(inputs, self.W)
            action = self._select_action(self._softmax(outputs, self.t))
            e = np.zeros(self.W.shape)
            i = 0 # Escape latency
            while True:
                # Carry out action
                self.mountain_car.apply_force(action - 1)
                self.mountain_car.simulate_timesteps(100, 0.01)

                inputs_ = self._compute_inputs(self.mountain_car.x, self.mountain_car.x_d)
                outputs_ = self._compute_outputs(inputs_, self.W)
                action_ = self._select_action(self._softmax(outputs_, self.t))

                delta = eta * (self.mountain_car.R + gamma*outputs_[action_] - outputs[action])

                # Update eligibility trace
                e = e * gamma * decay
                e[:, action] = e[:, action] + inputs

                # Update weights
                self.W = self.W + delta * e

                # When the agent obtains the reward, a new trial is started.
                # The 10000 iterations limit is there to prevent infinite loops
                # in case the agent gets stuck in a local minimum.
                if self.mountain_car.R > 0 or i == 10000:
                    print('Reward')
                    self.mountain_car.reset()
                    self.t *= t_decay
                    print('Trial: ', trial, ' - Temperature: ', self.t, ' - Time to escape: ', i)
                    self.results.append(i)
                    break

                state = self.mountain_car.x, self.mountain_car.x_d
                action = action_
                inputs = inputs_
                outputs = outputs_
                i += 1
        return self.results
    
    def _compute_inputs(self, x, v):
        ''' Compute the neural network inputs from the current state of the agent. ''' 
        out = np.empty((self.x_n, self.v_n))
        for i, x_c in enumerate(self.x_centers):
            for j, v_c in enumerate(self.v_centers):
                out[i, j] = np.exp(-(x_c - x)**2 / self.x_sigma**2 - (v_c - v)**2 / self.v_sigma**2)
        return out.reshape(-1)

    def _compute_outputs(self, inputs, W):
        ''' Compute the Q(s,a) for each a '''
        # Computes the Q-values
        return inputs.dot(W)

    def _softmax(self, values, t):
        ''' Softmax function, with temperature parameter '''
        # Numerically stable implementation of the softmax function
        if t >= 1e6: # Infinity (random actions)
            return np.ones(values.size)/values.size
        elif t <= 1e-6: # Zero (action associated with the highest Q-value)
            e = np.zeros(values.size)
            e[np.argmax(values)] = 1.0
            return e
        else:
            e = values / t
            return np.exp(e - scipy.misc.logsumexp(e))

    def _select_action(self, p):
        ''' Sample a value from the given probability distribution '''
        return self.rnd.choice(p.size, p=p)