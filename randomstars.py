from random import random, uniform, randint, gauss
from math import sin, cos, pi, sqrt, acos

from pygame import image

from star import Star

class RandomStars(object):
    """Randomly generated star data for visible stars around a sun like ours."""
    def __init__(self):
        """Initialize the random star data."""
        
        self._hist = image.load('hist.pgm')
        
        self.stars = [self.getrandom() for i in range(6000)]
        """List of stars."""

        z = uniform(-1, 1)
        theta = uniform(0, 2 * pi)
        r = sqrt(1 - z ** 2)
        u = r * cos(theta), r * sin(theta), z
        
        self.rotation = uniform(0, 2 * pi), u
        """Uniformly selected rotation from galactic plane."""

    def getrandom(self):
        """Get a randomly generated star."""

        w, h = self._hist.get_size()
        while True:
            c = uniform(-0.274, 3.271)
            m = uniform(-15.329, 7.49)

            x = int((c + 0.274) * w/(3.271+0.274))
            y = int((m + 15.329) * h/(7.49+15.329))
            
            if randint(0, 255) > self._hist.get_at((x,y))[0]:

                r = uniform(1, 3.26/pow(10, (m - 6)/5 - 1))
                if random() > sqrt((x*y)/float(w*h)):
                    theta = acos(max(-1, min(1, gauss(0, 0.125))))
                else:
                    theta = acos(uniform(-1, 1))
                phi = uniform(0, 2 * pi)
                return Star((r, theta, phi), c, m, abs(pi/2 - theta))
