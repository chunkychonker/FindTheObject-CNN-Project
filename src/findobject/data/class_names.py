"""Human-readable names for the 43 GTSRB class ids."""

from findobject.data.gtsrb import NUM_CLASSES

CLASS_NAMES: tuple[str, ...] = (
    "speed 20", "speed 30", "speed 50", "speed 60", "speed 70", "speed 80", "end speed 80", "speed 100",
    "speed 120", "no passing", "no passing trucks", "right-of-way", "priority road", "yield", "stop",
    "no vehicles", "trucks prohibited", "no entry", "general caution", "curve left", "curve right",
    "double curve", "bumpy road", "slippery road", "road narrows right", "road work", "traffic signals",
    "pedestrians", "children crossing", "bicycles crossing", "ice/snow", "wild animals", "end of limits",
    "turn right ahead", "turn left ahead", "ahead only", "straight or right", "straight or left",
    "keep right", "keep left", "roundabout", "end no passing", "end no passing trucks",
)

assert len(CLASS_NAMES) == NUM_CLASSES, "one name per class id"
