class Food():
    ID_counter = 0

    def __init__(self, x, y, r, color):
        self.x_pos = x
        self.y_pos = y
        self.radius = r
        self.color = color

        # give unique IDs to objects for debug purposes
        self.id = Food.ID_counter
        Food.ID_counter += 1

    def get_pos(self):
        return (self.x_pos, self.y_pos)
