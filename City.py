
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # get distance between this city and city param
    def distance(self, city):
        x_dis = abs(self.x - city.x)
        y_dis = abs(self.y - city.y)
        return (x_dis ** 2 + y_dis ** 2) ** 0.5

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
