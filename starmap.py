from math import *

import pygame
from pygame.locals import *

from quat import quat
from hipparcos import StarData
from randomstars import RandomStars

# Calculations involving great circles based on example code by Chris Veness:
# http://www.movable-type.co.uk/scripts/latlong.html
# Used under a Creative Commons license:
# http://creativecommons.org/licenses/by/3.0/

# Weighted average of points on a sphere uses algorithm by
# Samuel Buss and Jay Fillmore
# http://www.math.ucsd.edu/~sbuss/ResearchWeb/spheremean/paper.pdf

class Planet:
    def __init__(self, radius, tile_size, value):
        self.radius = float(radius)
        self.dtheta = tile_size / self.radius
        self.row_count = int(pi / self.dtheta)
        self.row_lengths = [l for l in
                            [int(2 * self.row_count * sin(row * self.dtheta) + 0.5)
                             for row in range(0,int(pi/self.dtheta))]
                            if l > 0]
        self.row_count = len(self.row_lengths)
        self.rows = [[value  for i in range(0, row_length)]
                     for row_length in self.row_lengths]
        self.max_row = max(self.row_lengths)
        self.row_offsets = [int((self.max_row - row_length)/2.0 + 0.5)
                            for row_length in self.row_lengths]

    def get_slope(self, row, nrow):
        return float(self.row_lengths[int(nrow)])/self.row_lengths[int(row)]

    def get_adjacent_row(self, row, column, nrow):
        if nrow >= 0 and nrow < self.row_count - 1 and self.row_lengths[int(nrow)]:
            m = self.get_slope(row, nrow)
            start = int(column * m)
            end = int((column + 1) * m) + 1
            if start < 0:
                # do wrapped portion on right and limit start
                for c in range (start, 0):
                    yield (nrow, self.row_lengths[int(nrow)] + c)
                start = 0
            if end > self.row_lengths[int(nrow)]:
                # do wrapped portion on left and limit end
                for c in range(self.row_lengths[int(nrow)], end):
                    yield (nrow, c - self.row_lengths[int(nrow)])
                end = self.row_lengths[int(nrow)]
            for c in range(start, end):
                yield (nrow, c)
    
    def adjacent(self, row, column):
        # wrap left
        if column > 0:
            yield (row, column-1)
        elif self.row_lengths[row] > 1:
            yield (row, self.row_lengths[int(row)]-1)

        # wrap right
        if column < self.row_lengths[int(row)]-1:
            yield (row, column+1)
        elif self.row_lengths[int(row)] > 1:
            yield (row, 0)

        # adjacent rows
        for nrow in (row-1,row+1):
            for adj in self.get_adjacent_row(row, column, nrow):
                yield adj

    def get_coordinates(self, row, column, size=None):
        size = size or (0,0)
        return (column + self.row_offsets[int(row)] - size[0]/2,
                row - size[1]/2)

    def get_coordinates_from_lat_lon(self, lat, lon, size=None):
        row = min(lat/-self.dtheta + self.row_count/2, self.row_count-1)
        column = (lon + pi) / (2 * pi) * self.row_lengths[int(row)]
        return self.get_coordinates(row, column, size)

    def get_row_column(self, x, y, size=None):
        size = size or (0,0)
        row = y + size[1]/2
        return (row,
                x - self.row_offsets[int(row)] + size[0]/2
                if 0 <= row < self.row_count else None)

    def in_projection(self, x, y, size=None):
        row, column = self.get_row_column(x, y, size)
        return (0 <= row < self.row_count and
                0 <= column < len(self.rows[int(row)]))

    def get_lat_lon(self, x, y, size=None):
        row, column = self.get_row_column(x, y, size)
        return (-(row - self.row_count/2) * self.dtheta,
                2 * pi * column/self.row_lengths[int(row)] - pi)

    def xy_bearing(self, x1, y1, size1, x2, y2, size2):
        lat1, lon1 = self.get_lat_lon(x1, y1, size1)
        lat2, lon2 = self.get_lat_lon(x2, y2, size2)

        return self.bearing(lat1, lon1, lat2, lon2)

    def distance(self, x1, y1, size1, x2, y2, size2):
        lat1, lon1 = self.get_lat_lon(x1, y1, size1)
        lat2, lon2 = self.get_lat_lon(x2, y2, size2)

        return acos(sin(lat1)*sin(lat2) +
                    cos(lat1)*cos(lat2)*cos(lon2-lon1))

    def midpoint(self, x1, y1, size1, x2, y2, size2, size=None):
        lat1, lon1 = self.get_lat_lon(x1, y1, size1)
        lat2, lon2 = self.get_lat_lon(x2, y2, size2)

        bx = cos(lat2) * cos(lon2 - lon1)
        by = cos(lat2) * sin(lon2 - lon1)
        latm = atan2(sin(lat1) + sin(lat2),
                     sqrt((cos(lat1) + bx)*(cos(lat1) + bx) + by*by))
        lonm = lon1 + atan2(by, cos(lat1) + bx)
        if lonm > pi:
            lonm -= 2*pi
        if lonm < -pi:
            lonm += 2*pi
        
        return self.get_coordinates_from_lat_lon(latm, lonm, size)

    def xy_to_vector(self, x, y, size=None):
        lat, lon = self.get_lat_lon(x, y, size)
        cos_lat = cos(lat)
        return (cos_lat * cos(lon),
                cos_lat * sin(lon),
                sin(lat))

    def vector_to_xy(self, v, size=None):
        lat = atan2(v[2], sqrt(v[0]*v[0] + v[1]*v[1]))
        lon = atan2(v[1], v[0])
        return self.get_coordinates_from_lat_lon(lat, lon, size)

    def magnitude(self, p):
        return sqrt(sum([x*x for x in p]))

    def weighted_average(self, xy_points, weights, sizes, size=None):
        ps = [self.xy_to_vector(xy_points[i][0], xy_points[i][1], sizes[i])
              for i in range(len(xy_points))]
        return self.vector_to_xy(self.vector_weighted_average(ps, weights), size)

    def vector_weighted_average(self, ps, weights):
        add_vectors = lambda p1, p2: tuple([p1[i]+p2[i] for i in range(len(p1))])
        vector_diff = lambda p1, p2: tuple([p1[i]-p2[i] for i in range(len(p1))])
        weighted_sum = reduce(
            add_vectors,
            [tuple([weights[i] * c for c in ps[i]]) for i in range(len(ps))])
        q = tuple([x / self.magnitude(weighted_sum) for x in weighted_sum])
        last_u = None
        while True:
            p_stars = []
            for p in ps:
                cos_th = sum([p[i] * q[i] for i in range(len(p))])
                th = acos(cos_th)
                p_stars.append(tuple([c * th / sin(th) for c in p]))
            u = reduce(
                add_vectors,
                [tuple([weights[i] * c
                        for c in vector_diff(p_stars[i], q)])
                 for i in range(len(p_stars))])
            mag_u = self.magnitude(u)
            if mag_u == last_u:
                return q
            else:
                last_u = mag_u

    def bearing(self, lat1, lon1, lat2, lon2):
        return atan2(sin(lon2-lon1) * cos(lat2),
                     cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(lon2-lon1))

    def apply_bearing(self, d, theta, x, y, size=None):
        lat, lon = self.get_lat_lon(x, y, size)

        da = d/self.radius
        lat2 = asin(sin(lat)*cos(da) +
                    cos(lat)*sin(da)*cos(theta))
        lon2 = lon + atan2(sin(theta)*sin(da)*cos(lat),
                           cos(da) - sin(lat)*sin(lat2))
        if lon2 > pi:
            lon2 -= 2*pi
        if lon2 < -pi:
            lon2 += 2*pi

        return (((self.bearing(lat2,lon2,lat,lon) + pi) % (2*pi),) +
                self.get_coordinates_from_lat_lon(lat2, lon2, size))

    def rotate(self, p, axis, theta):
        # http://inside.mines.edu/~gmurray/ArbitraryAxisRotation/ArbitraryAxisRotation.html
        L2 = sum([i*i for i in axis])
        return array([(axis[0]*sum(p * axis) +
                       (p[0]*(axis[1]*axis[1]+axis[2]*axis[2]) +
                        axis[0]*(-axis[1]*p[1]-axis[2]*p[2])) * cos(theta) +
                       sqrt(L2)*(-axis[2]*p[1]+axis[1]*p[2]) * sin(theta))/L2,
                      (axis[1]*sum(p * axis) +
                       (p[1]*(axis[0]*axis[0]+axis[2]*axis[2]) +
                        axis[1]*(-axis[0]*p[0]-axis[2]*p[2])) * cos(theta) +
                       sqrt(L2)*(axis[2]*p[0]-axis[0]*p[2]) * sin(theta))/L2,
                      (axis[2]*sum(p * axis) +
                       (p[2]*(axis[0]*axis[0]+axis[1]*axis[1]) +
                        axis[2]*(-axis[0]*p[0]-axis[1]*p[1])) * cos(theta) +
                       sqrt(L2)*(-axis[1]*p[0]+axis[0]*p[1]) * sin(theta))/L2])
                       

    def apply_velocity(self, p, v):
        d = norm(v)

        if d:
            axis = cross(p, v)
            axis = axis / norm(axis)

            return tuple([self.rotate(i, axis, d) for i in (p,v)])
        else:
            return p, v

    def repel_from_point(self, p, q):
        diff = q - p
        dist = math.acos(dot(q, p))
        return (pi - dist) * (dot(diff, p) * p - diff)

    def project_on_plane(self, v, n):
        return v - dot(v, n) * n

    def apply_heading(self, v, theta, x, y, size=None):
        row, column = self.get_row_column(x, y, size)
        ntheta = theta

        # vertical component
        vy = v * sin(theta)
        nrow = row + vy
        if nrow < 0:
            nrow = -nrow
            ntheta = (theta + pi) % (2 * pi)
        elif nrow >= self.row_count:
            nrow = self.row_count - (nrow - self.row_count)
            ntheta = (theta + pi) % (2 * pi)
        m = self.get_slope(row, nrow)
        
        # horizontal component
        vx = v * cos(theta)
        ncolumn = (column + vx) * m
        if theta != ntheta:
            if ncolumn > self.row_lengths[int(nrow)]/2:
                ncolumn -= self.row_lengths[int(nrow)]/2
            else:
                ncolumn += self.row_lengths[int(nrow)]/2
        if ncolumn < 0:
            ncolumn += self.row_lengths[int(nrow)]
        elif ncolumn > self.row_lengths[int(nrow)] - 1:
            ncolumn -= self.row_lengths[int(nrow)]
        x, y = self.get_coordinates(nrow, ncolumn, size)
        return ntheta, x, y

def offset(p, dx, dy):
    x,y = p
    return x + dx, y + dy

def rotate(p, loc, th):
    x,y = [p[i]-loc[i] for i in range(2)]
    s,c = sin(th),cos(th)
    return x*c - y*s + loc[0], x*s + y*c + loc[1]

def defloat(p):
    x,y = p
    return int(x+0.5), int(y+0.5)

def ranges(ps):
    return [[f([p[i] for p in ps])
             for i in range(2)]
            for f in (min,max)]

class Display:

    def main_loop(self, realdata):

        planet = Planet(200,1,0)

        pygame.init()    

        screen = pygame.display.set_mode((planet.max_row,planet.row_count),
                                         HWSURFACE)
        pygame.display.set_caption('Star Map')

        background = pygame.Surface(screen.get_size())
        background.fill((128,128,128))

        def in_bounds(x,y):
            return (x > planet.row_offsets[y] and
                    x < planet.row_offsets[y] + planet.row_lengths[y])

        background.lock()
        for y in range(0, screen.get_height()):
            for x in range(0, screen.get_width()):
                if in_bounds(x,y):
                    background.set_at((x,y), (0,0,0))
        background.unlock()

        def color(index):
            if index < 0.25:
                s = int(255 * (index + 0.5)/0.75)
                return s, s, 255
            elif index < 1:
                return 255, 255, int(255 * (1 - index)/0.75)
            else:
                return 255, int(255 * (2 - min(2, index))), 0

        def magnitude(color, magnitude, distance):
            apparent = magnitude - 5 * (log10(distance/3.26) + 1)
            scale = 1 - (apparent + 40)/45.0
            scale = min(1, scale)
            return [int(c * scale) for c in color]

        data = StarData() if realdata else RandomStars()

        rot_th, rot_u = data.rotation
        rot = quat(rot_u[0] * sin(rot_th/2),
                   rot_u[1] * sin(rot_th/2),
                   rot_u[2] * sin(rot_th/2),
                   cos(rot_th/2))
        
        for s in data.stars:
            r, theta, phi = s.location

            sin_th = sin(theta)

            v = sin_th * cos(phi), sin_th * sin(phi), cos(theta)
            
            q = quat(0, *v)
            v = ((rot*q)/rot).q[1:4]
            
            x,y = planet.vector_to_xy(v)

            background.set_at((int(x),int(y)),
                              magnitude(color(s.color), s.magnitude, r))

        screen.blit(background, (0,0))
            
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == QUIT:
                    done = True
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        done = True
            
            pygame.display.flip()

if __name__ == '__main__':
    from sys import argv
    realdata = len(argv) > 1 and argv[1] == 'r'
    Display().main_loop(realdata)
