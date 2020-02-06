#!/usr/bin/python3

import numpy as np
import functools
from numbers import Number

def check_type(func):
    @functools.wraps(func)
    def wrapper_check_type(*args):
        z = len(args)
        if z > 2:
            raise TypeError(f"Too many arguments; expected 1 or 2, but got {z}")
        index = 1 if z == 2 else 0
        arg_name = func.__code__.co_varnames[index]
        if isinstance(args[index], (Number, Angle, np.ndarray)):
            return func(*args)
        else:
            err_msg = f"\'{arg_name}\' must be an int, float, Angle, or ndarray"
            raise NotImplementedError(err_msg)
    return wrapper_check_type      

class Angle():
    """ 
    Convenience class that stores an angle (float in radians) and forces it to 
    stay wrapped between pi and -pi. 
    """  
    @staticmethod    
    def wrap_value(newang):
        return (newang-np.pi) % (2*np.pi) - np.pi
    
    @classmethod
    def wrap_init(cls, newang):
        return cls.wrap_value(newang) 

    @classmethod
    def wrap(cls, newang):
        result = cls.wrap_value(newang)
        return cls(result)

    def __init__(self, input_ang=0.0, deg=False):

        if isinstance(input_ang, self.__class__):
            self.angle = input_ang.angle
            return
        elif isinstance(input_ang, Number):
            if deg:
                input_ang = np.radians(input_ang)
            self.angle = self.wrap_init(input_ang)
        else:
            raise TypeError(f"input_ang must be an int or float")


    def __array__(self):
        return self.angle

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            scalars = []
            for input in inputs:
                if isinstance(input, Number):
                    scalars.append(input)
                elif isinstance(input, self.__class__):
                    scalars.append(input.angle)
                elif isinstance(input, np.ndarray):
                    scalars.append(input)
                else:
                    return NotImplemented
            return self.__class__(ufunc(*scalars, **kwargs))
        else:
            return NotImplemented

    @check_type
    def __add__(self, angle2):
        if isinstance(angle2, self.__class__):
            newang = self.angle + angle2.angle
        elif isinstance(angle2, np.ndarray):
            newang = self.__array_ufunc__(np.add, '__call__', self, angle2)
        else:
            newang = self.angle + angle2
        return self.wrap(newang)

    @check_type
    def __sub__(self, angle2):
        if isinstance(angle2, self.__class__):
            newang = self.angle - angle2.angle
        elif isinstance(angle2, np.ndarray):
            newang = self.__array_ufunc__(np.subtract, '__call__', self, angle2)
        else:
            newang = self.angle - angle2
        return self.wrap(newang)
    @check_type
    def __mul__(self, angle2):
        if isinstance(angle2, self.__class__):
            newang = self.angle * angle2.angle
        elif isinstance(angle2, np.ndarray):
            newang = self.__array_ufunc__(np.multiply, '__call__', self, angle2)
        else:
            newang = self.angle * angle2
        return self.wrap(newang)
    @check_type
    def __truediv__(self, angle2):
        if isinstance(angle2, self.__class__):
            newang = self.angle / angle2.angle
        elif isinstance(angle2, np.ndarray):
            newang = self.__array_ufunc__(np.truedivide, '__call__', self, angle2)
        else:
            newang = self.angle / angle2
        return self.wrap(newang)

    # Reverse operators
    @check_type
    def __radd__(self, angle2):
        return self.__add__(angle2)
    @check_type
    def __rsub__(self, angle2):
        Angle2 = self.__class__(angle2)
        return Angle2.__sub__(self.angle)
    @check_type
    def __rmul__(self, angle2):
        return self.__mul__(angle2)
    @check_type
    def __rtruediv__(self, angle2):
        Angle2 = self.__class__(angle2)
        return Angle2.__truediv__(self.angle)

    def __neg__(self):
        return self.__class__(-self.angle)
    
    def __str__(self):
        return str(self.angle)

    def deg(self):
        return np.degrees(self.angle)
    
if __name__ == '__main__':
    ang1 = Angle(370, True)
    ang2 = Angle(-160, True)
    ang3 = Angle(824, True)
    
    a1 = np.array([ang1, ang2, ang3])
    a2 = np.array([ang1, ang3, ang2])
    a3 = a1 + a2
    c = ang1 * ang2
    q = ang3 + ang1 / ang2
    
    print(ang3)
    print(ang3.deg())