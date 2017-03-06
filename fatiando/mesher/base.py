"""
Base classes for geometries and meshes.
"""
from __future__ import division, print_function
from future.utils import with_metaclass
from future.builtins import object, super
from abc import ABCMeta, abstractmethod
import copy
from operator import mul
from functools import reduce


class GeometricElement(object):
    """
    Base class for all geometric elements.

    Defines the physical property dictionary ``props`` and the ``addprop``
    method for adding new ones.
    """

    def __init__(self, props):
        if props is None:
            self.props = dict()
        else:
            self.props = copy.copy(props)

    def addprop(self, prop, value):
        """
        Add a physical property to this geometric element.

        If it already has the property, the given value will overwrite the
        existing one.

        Parameters:

        * prop : str
            Name of the physical property.
        * value : float
            The value of this physical property.

        """
        self.props[prop] = value

    def copy(self):
        "Return a deep copy of the current instance."
        return copy.deepcopy(self)


class RegularMesh(with_metaclass(ABCMeta, GeometricElement)):
    """
    Base class for all regular meshes.

    Defines the basic methods for iteration and common attributes of all
    meshes.
    """

    def __init__(self, bounds, shape, props=None):
        super().__init__(props)
        self.bounds = bounds
        self.shape = shape
        self.size = reduce(mul, shape)
        # Used as the index when iterating over the mesh
        self._index = 0
        # List of masked elements that will be ignored during iteration
        self.mask = []

    def __len__(self):
        return self.size

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= self.size:
            raise StopIteration
        element = self.__getitem__(self._index)
        self._index += 1
        return element

    def __getitem__(self, index):
        if index < 0:
            # To walk backwards in the mesh
            index = self.size + index
        if index in self.mask:
            return None
        element = self._get_element(index)
        props = dict((p, self.props[p][index]) for p in self.props)
        element.props = props
        return element

    @abstractmethod
    def _get_element(self, index):
        """
        Return the element of the mesh corresponding to a given index.
        """
        pass

    def addprop(self, prop, values):
        """
        Add physical property values to the elements of the mesh.

        Different physical properties of the mesh are stored in the
        ``mesh.props`` dictionary.

        Parameters:

        * prop : str
            Name of the physical property
        * values : list or array
            The value of this physical property in each element of the mesh.
            See the mesh class docstring for the order of elements.

        """
        self.props[prop] = values
