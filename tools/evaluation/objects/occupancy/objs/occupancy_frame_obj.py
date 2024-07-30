class OccupancyFrame:
    def __init__(self, data, ts):
        self.data = data
        self.ts = ts

    def get_labeled_voxel(self):
        return self.data
