""" This module contains classes that form the basis for part 2 of the assignment

    Refer the the coursework assignment for instructions on how to complete this part.
"""
import math
import statistics


class Point:
    """"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def set_x(self, x):
        self.x = x

    def get_y(self):
        return self.y

    def set_y(self, y):
        self.y = y

    def distance(self, point):
        # Calculate the Euclidean distance
        return math.sqrt((abs(self.x - point.x) ** 2) + (abs(self.y - point.y) ** 2))

    def equals(self, point):
        return self.distance(point) < Shape.TOLERANCE


class Shape:
    """This class is a convenient place to store the tolerance variable"""
    TOLERANCE = 1.0e-6


class Circle:

    def __init__(self, centre, radius):
        self.centre = centre
        self.radius = radius

    def get_centre(self):
        return self.centre

    def set_centre(self, centre):
        self.centre = centre

    def get_radius(self):
        return self.radius

    def set_radius(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * (self.radius ** 2)

    def compare(self, shape):
        this_area = self.area()
        other_area = shape.area()
        if this_area < other_area:
            return -1
        elif this_area > other_area:
            return 1
        return 0

    def envelops(self, shape):
        if isinstance(shape, Square):  # If the inner shape is a square
            #  Check if all the vertices of the inscribed square are within the boundaries of the circle
            top_left = shape.get_top_left()
            top_right = Point(shape.get_top_left().get_x() + shape.get_side_length(), shape.get_top_left().get_y())
            bottom_left = Point(shape.get_top_left().get_x(), shape.get_top_left().get_y() - shape.get_side_length())
            bottom_right = Point(shape.get_top_left().get_x() + shape.get_side_length(), shape.get_top_left().get_y() - shape.get_side_length())
            if (self.get_centre().distance(top_left) < self.get_radius() and
                    self.get_centre().distance(top_right) < self.get_radius() and
                    self.get_centre().distance(bottom_left) < self.get_radius() and
                    self.get_centre().distance(bottom_right) < self.get_radius()):
                return True
        elif isinstance(shape, Circle): # If the inner shape is a square
            dist = self.get_centre().distance(shape.get_centre())
            if self.get_radius() > shape.get_radius() + dist:
                return True
        return False

    def equals(self, circle):
        return self.centre.equals(circle.centre) and abs(self.radius - circle.radius) < Shape.TOLERANCE

    def __str__(self):
        return 'This circle has its centre at ({centre_x},{centre_y}) and a radius of {radius}.'.format(centre_x=self.centre.get_x(), centre_y=self.centre.get_y(), radius=self.get_radius())


class Square:

    def __init__(self, top_left, length):
        self.top_left = top_left
        self.side_length = length

    def set_top_left(self, top_left):
        self.top_left = top_left

    def get_top_left(self):
        return self.top_left

    def set_side_length(self, length):
        self.side_length = length

    def get_side_length(self):
        return self.side_length

    def area(self):
        return self.side_length ** 2

    def compare(self, shape):
        this_area = self.area()
        other_area = shape.area()
        if this_area < other_area:
            return -1
        elif this_area > other_area:
            return 1
        return 0

    def envelops(self, shape):
        if isinstance(shape, Square): # If the inner shape is a square
            if (shape.get_top_left().get_x() > self.get_top_left().get_x() and
                    shape.get_top_left().get_x() + shape.get_side_length() < self.get_top_left().get_x() + self.get_side_length() and
                    shape.get_top_left().get_y() < self.get_top_left().get_y() and
                    shape.get_top_left().get_y() - shape.get_side_length() > self.get_top_left().get_y() - self.get_side_length()):
                return True
        elif isinstance(shape, Circle): # If the inner shape is a circle
            if (shape.get_centre().get_x() - shape.get_radius() > self.get_top_left().get_x() and
                    shape.get_centre().get_x() + shape.get_radius() < self.get_top_left().get_x() + self.get_side_length() and
                    shape.get_centre().get_y() + shape.get_radius() < self.get_top_left().get_y() and
                    shape.get_centre().get_y() - shape.get_radius() > self.get_top_left().get_y() - self.get_side_length()):
                return True
        return False

    def equals(self, square):
        return self.top_left.equals(square.top_left) and abs(self.side_length - square.side_length) < Shape.TOLERANCE

    def __str__(self):
        return 'This square’s top left corner is at ({top_left_x},{top_left_y}) and the side length is {side_length}.'.format(top_left_x=self.top_left.get_x(), top_left_y=self.top_left.get_y(), side_length=self.get_side_length())


class Assignment:

    def __init__(self):
        self.squares = []
        self.circles = []

    def analyse(self, filename):
        """ Process the file here """
        # Open the input file and read all the lines
        with open(filename) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            if line[0] == 'square':
                tmp_side_length = float(line[3])
                if tmp_side_length < Shape.TOLERANCE:  # Check if it is singular
                    continue
                tmp_top_left = Point(float(line[1]), float(line[2]))
                tmp_square = Square(tmp_top_left, tmp_side_length)
                self.squares.append(tmp_square)

            elif line[0] == 'circle':
                tmp_radius = float(line[3])
                if tmp_radius < Shape.TOLERANCE:  # Check if it is singular
                    continue
                tmp_centre = Point(float(line[1]), float(line[2]))
                tmp_circle = Circle(tmp_centre, tmp_radius)
                self.circles.append(tmp_circle)

    def shape_count(self):
        return self.square_count() + self.circle_count()

    def circle_count(self):
        return len(self.circles)

    def square_count(self):
        return len(self.squares)

    def max_circle_area(self):
        areas = list(map(Circle.area, self.circles))
        return max(areas)

    def min_circle_area(self):
        areas = list(map(Circle.area, self.circles))
        return min(areas)

    def max_square_area(self):
        areas = list(map(Square.area, self.squares))
        return max(areas)

    def min_square_area(self):
        areas = list(map(Square.area, self.squares))
        return min(areas)

    def mean_circle_area(self):
        areas = list(map(Circle.area, self.circles))
        return sum(areas)/len(areas)

    def mean_square_area(self):
        areas = list(map(Square.area, self.squares))
        return statistics.mean(areas)

    def std_dev_circle_area(self):
        areas = list(map(Circle.area, self.circles))
        return statistics.stdev(areas)

    def std_dev_square_area(self):
        areas = list(map(Square.area, self.squares))
        return statistics.stdev(areas)

    def median_circle_area(self):
        areas = list(map(Circle.area, self.circles))
        return statistics.median(areas)

    def median_square_area(self):
        areas = list(map(Square.area, self.squares))
        return statistics.median(areas)


if __name__ == "__main__":
    # You should add your own code here to test your work
    print("=== Testing Part 2 ===")
    assignment = Assignment()
    assignment.analyse("SmallShapeTest.data")
    for circle in assignment.circles:
        print(circle)
    for square in assignment.squares:
        print(square)
    print('shape_count')
    print(assignment.shape_count())
    print('circle_count')
    print(assignment.circle_count())
    print('square_count')
    print(assignment.square_count())
    print('max_circle_area')
    print(assignment.max_circle_area())
    print('min_circle_area')
    print(assignment.min_circle_area())
    print('max_square_area')
    print(assignment.max_square_area())
    print('min_square_area')
    print(assignment.min_square_area())
    print('mean_circle_area')
    print(assignment.mean_circle_area())
    print('mean_square_area')
    print(assignment.mean_square_area())
    print('std_dev_circle_area')
    print(assignment.std_dev_circle_area())
    print('std_dev_square_area')
    print(assignment.std_dev_square_area())
    print('median_circle_area')
    print(assignment.median_circle_area())
    print('median_square_area')
    print(assignment.median_square_area())
    print(assignment.circles[0].envelops(assignment.squares[0]))
