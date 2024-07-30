

class TrafficLightEnum:
    class Color:
        Unknown = "Unknown"
        Red = "Red"
        Yellow = "Yellow"
        Green = "Green"
        Black = "Black"

    onboard_color_enum = {0: Color.Unknown,
                          1: Color.Red,
                          2: Color.Yellow,
                          3: Color.Green,
                          4: Color.Black}

    super_color_enum = {
                        "Unknown": Color.Unknown, "unknown": Color.Unknown, "UNKNOWN": Color.Unknown,

                        "Red": Color.Red, "red": Color.Red, "RED": Color.Red,

                        "Yellow": Color.Yellow, "yellow": Color.Yellow, "YELLOW": Color.Yellow,

                        "Green": Color.Green, "green": Color.Green, "GREEN": Color.Green,

                        "Black": Color.Black, "black": Color.Black, "BLACK": Color.Black
                        }


class Attr:
    ts = "ts"
    light_shape = "light_shape"
    color = "color"
    score = "score"
    xmin = "xmin"
    ymin = "ymin"
    xmax = "xmax"
    ymax = "ymax"
    width = "width"
    height = "height"
    corners_2d = "corners_2d"
    infer_stage = "infer_stage"
    roi_xmin = "roi_xmin"
    roi_ymin = "roi_ymin"
    roi_xmax = "roi_xmax"
    roi_ymax = "roi_ymax"
    roi_width = "roi_width"
    roi_height = "roi_height"
