from unittest import TestCase

import numpy as np

from data.rectangles import Rectangle


class TestRectangle(TestCase):
    def test__intersect(self):

        # patch contains the complete foreground_object annotation
        patch = Rectangle(100, 100, 200, 200)
        foreground_object = Rectangle(150, 150, 175, 175)
        np.testing.assert_(patch.intersects(foreground_object))
        np.testing.assert_(foreground_object.intersects(patch))

        # foreground_object annotation contains the complete patch
        patch = Rectangle(150, 150, 175, 175)
        foreground_object = Rectangle(100, 100, 200, 200)
        np.testing.assert_(patch.intersects(foreground_object))
        np.testing.assert_(foreground_object.intersects(patch))

        # partial overlap in both dimensions
        patch = Rectangle(100, 100, 200, 200)
        foreground_object = Rectangle(150, 150, 250, 250)
        np.testing.assert_(patch.intersects(foreground_object))
        np.testing.assert_(foreground_object.intersects(patch))

        # complete overlap in y-dimension only
        patch = Rectangle(100, 100, 200, 200)
        foreground_object = Rectangle(150, 0, 175, 250)
        np.testing.assert_(patch.intersects(foreground_object))
        np.testing.assert_(foreground_object.intersects(patch))

        # complete overlap in x-dimension only
        patch = Rectangle(100, 100, 200, 200)
        foreground_object = Rectangle(50, 150, 250, 175)
        np.testing.assert_(patch.intersects(foreground_object))
        np.testing.assert_(foreground_object.intersects(patch))
