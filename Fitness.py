from City import City


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def route_distance(self):
        if self.distance == 0:
            path_distance = 0
            for i in range(len(self.route)):
                from_city_coords = self.route[i]

                if i + 1 < len(self.route):
                    to_city_coords = self.route[i + 1]
                else:
                    to_city_coords = self.route[0]

                from_city = City(x=from_city_coords[0], y=from_city_coords[1])
                to_city = City(x=to_city_coords[0], y=to_city_coords[1])

                path_distance += from_city.distance(to_city)
            self.distance = path_distance
        return self.distance


    def route_fitness(self):
        if self.fitness == 0.0:
            self.fitness = 1 / float(self.route_distance())
        return self.fitness



