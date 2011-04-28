import math

from hipparcos import StarData
from randomstars import RandomStars

if __name__ == '__main__':
    from matplotlib import pyplot, axes
    import numpy

    from sys import argv

    d = StarData()
    r = RandomStars()

    pyplot.subplot(321)
    pyplot.scatter([s.color for s in d.stars],
                   [s.magnitude for s in d.stars],
                   marker='+',
                   alpha=0.025)

    pyplot.subplot(322)
    pyplot.scatter([s.color for s in r.stars],
                   [s.magnitude for s in r.stars],
                   marker='+',
                   alpha=0.025)

    pyplot.subplot(323)
    H, xedges, yedges = numpy.histogram2d(
        [s.color for s in d.stars],
        [s.magnitude for s in d.stars],
        normed=True,
        bins=(100,100))
    
    extent = [xedges[0], xedges[-1] * 10, yedges[0], yedges[-1]]                
    pyplot.imshow(H,
                  cmap=pyplot.cm.gray_r,
                  extent=extent,
                  interpolation='nearest')
    
    pyplot.subplot(324)
    H, xedges, yedges = numpy.histogram2d(
        [s.color for s in r.stars],
        [s.magnitude for s in r.stars],
        normed=True,
        bins=(100,100))
    
    extent = [xedges[0], xedges[-1] * 10, yedges[0], yedges[-1]]                
    pyplot.imshow(H,
                  cmap=pyplot.cm.gray_r,
                  extent=extent,
                  interpolation='nearest')

    from mpl_toolkits.mplot3d import axes3d

    axes = pyplot.subplot(325, projection='3d')
    axes.scatter([s.color for s in d.stars],
                 [s.magnitude for s in d.stars],
                 [s.offset for s in d.stars])

    axes = pyplot.subplot(326, projection='3d')
    axes.scatter([s.color for s in r.stars],
                 [s.magnitude for s in r.stars],
                 [s.offset for s in r.stars])

    pyplot.show()
