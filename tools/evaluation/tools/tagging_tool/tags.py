from enum import Enum, auto

class BaseTag:
    pass


class SameAsEgo(BaseTag):
    def __str__(self):
        return "Direction tag: SameAsEgo"

class Oncoming(BaseTag):
    def __str__(self):
        return "Direction tag: Oncoming"

class Direction:
    SameAsEgo = SameAsEgo
    Oncoming = "oncoming"


class Crossing(Direction):
    LeftCrossing = "left_crossing"
    RightCrossing = "right_crossing"




class Tags:
    def __contains__(self, item):
        pass


