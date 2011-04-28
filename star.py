class Star(object):
    """Characteristics of a single star."""
    def __init__(self, location, color, magnitude, offset):
        """Define a star with the given properties."""
        
        self.location = location
        """Location of the star in spherical coordinates,
           as an (r, theta, phi) tuple."""

        self.color = color
        """B-V color index."""
        
        self.magnitude = magnitude
        """The star's absolute magnitude."""
        
        self.offset = offset
        """Arc distance of the star from the great circle on the celestial
           sphere that represents the galactic plane."""
