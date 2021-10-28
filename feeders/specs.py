openpose_joints = {
 "head": 0,
 "right wrist": 10,
 "left wrist": 9,
 "right elbow": 8,
 "left elbow": 7,
 "right shoulder": 6,
 "left shoulder": 5,
 "jaw": 73,

 "left thumb end": 28,
 "left index end": 32,
 "left middle end": 36,
 "left ring end": 40,
 "left little end": 44,

 "left little knuckle": 41,
 "left little joint 1": 42,
 "left little joint 2": 43,
 "left ring knuckle": 37,
 "left ring joint 1": 38,
 "left ring joint 2": 39,
 "left middle knuckle": 33,
 "left middle joint 1": 34,
 "left middle joint 2": 35,
 "left index knuckle": 29,
 "left index joint 1": 30,
 "left index joint 2": 31,
 "left thumb knuckle": 25,
 "left thumb joint 1": 26,
 "left thumb joint 2": 27,

 "right thumb end": 48,
 "right index end": 52,
 "right middle end": 56,
 "right ring end": 60,
 "right little end": 64,

 "right little knuckle": 61,
 "right little joint 1": 62,
 "right little joint 2": 63,
 "right ring knuckle": 57,
 "right ring joint 1": 58,
 "right ring joint 2": 59,
 "right middle knuckle": 53,
 "right middle joint 1": 54,
 "right middle joint 2": 55,
 "right index knuckle": 49,
 "right index joint 1": 50,
 "right index joint 2": 51,
 "right thumb knuckle": 45,
 "right thumb joint 1": 46,
 "right thumb joint 2": 47,

 "right eye": 2,
 "left eye": 1,
 "right ear": 4,
 "left ear": 3,
 }

holistic_joints = {
 "head": 0,
 "right wrist": 16,
 "left wrist": 15,
 "right elbow": 14,
 "left elbow": 13,
 "right shoulder": 12,
 "left shoulder": 11,
 "jaw": 185,

 "left thumb end": 526,
 "left index end": 530,
 "left middle end": 534,
 "left ring end": 538,
 "left little end": 542,

 "left little knuckle": 539,
 "left little joint 1": 540,
 "left little joint 2": 541,
 "left ring knuckle": 535,
 "left ring joint 1": 536,
 "left ring joint 2": 537,
 "left middle knuckle": 531,
 "left middle joint 1": 532,
 "left middle joint 2": 533,
 "left index knuckle": 527,
 "left index joint 1": 528,
 "left index joint 2": 529,
 "left thumb knuckle": 523,
 "left thumb joint 1": 524,
 "left thumb joint 2": 525,

 "right thumb end": 505,
 "right index end": 509,
 "right middle end": 513,
 "right ring end": 517,
 "right little end": 521,

 "right little knuckle": 518,
 "right little joint 1": 519,
 "right little joint 2": 520,
 "right ring knuckle": 514,
 "right ring joint 1": 515,
 "right ring joint 2": 516,
 "right middle knuckle": 510,
 "right middle joint 1": 511,
 "right middle joint 2": 512,
 "right index knuckle": 506,
 "right index joint 1": 507,
 "right index joint 2": 508,
 "right thumb knuckle": 502,
 "right thumb joint 1": 503,
 "right thumb joint 2": 504,

 "right eye": 5,
 "left eye": 2,
 "right ear": 8,
 "left ear": 7,
 }

joint_types = {"openpose": openpose_joints,
               "holistic": holistic_joints}


ears = (
    "right ear",
    "left ear",
)
upper_body = (
    "head",
    "jaw",
    "right wrist",
    "left wrist",
    "right elbow",
    "left elbow",
    "right shoulder",
    "left shoulder",
) + ears
left_hand = (
    "left thumb end",
    "left index end",
    "left middle end",
    "left ring end",
    "left little end",

    #"left little knuckle",
    "left little joint 1",
    #"left little joint 2",
    #"left ring knuckle",
    "left ring joint 1",
    #"left ring joint 2",
    #"left middle knuckle",
    "left middle joint 1",
    #"left middle joint 2",
    #"left index knuckle",
    "left index joint 1",
    #"left index joint 2",
    #"left thumb knuckle",
    "left thumb joint 1",
    #"left thumb joint 2",
)
right_hand = (
    "right thumb end",
    "right index end",
    "right middle end",
    "right ring end",
    "right little end",

    #"right little knuckle",
    "right little joint 1",
    #"right little joint 2",
    #"right ring knuckle",
    "right ring joint 1",
    #"right ring joint 2",
    #"right middle knuckle",
    "right middle joint 1",
    #"right middle joint 2",
    #"right index knuckle",
    "right index joint 1",
    #"right index joint 2",
    #"right thumb knuckle",
    "right thumb joint 1",
    #"right thumb joint 2",
)


POINTS = {
    "SMPL": upper_body,# + ears,
    "SMPLH": upper_body + left_hand + right_hand,# + ears,
    "MANO":  left_hand + right_hand, #("head",) +
    "smpl": upper_body,
    "smplh": upper_body + left_hand + right_hand,# + ears,
    "mano": left_hand + right_hand,#("head",) +
    "ears": ("left ear"),
}

OUTPUT_SIZES = {
    "SMPL": 24,
    "SMPLH": 73,
    "MANO": 144
}

right_arm_chain = ["right ear",
    "head", "jaw",
] + ["right " + name for name in ("shoulder", "elbow", "wrist")]
left_arm_chain = ["left ear",
    "head", "jaw",
] + ["left " + name for name in ("shoulder", "elbow", "wrist")]
upper_body_chains = [right_arm_chain, left_arm_chain]

finger_chain = (
    "knuckle",
    "joint 1",
    "joint 2",
    "end"
)
fingers = (
    " thumb ",
    " index ",
    " middle ",
    " ring ",
    " little "
)

right_hand_chains = [["right wrist"] + ["right" + finger + name for name in finger_chain] for finger in fingers]
left_hand_chains = [["left wrist"] + ["left" + finger + name for name in finger_chain] for finger in fingers]


name_neighbours = (
    ('head', 'jaw'),
    ('jaw', "right shoulder"),
    ("jaw", "left shoulder"),
    #  right
    ("right shoulder", "right elbow"),
    ("right elbow", "right wrist"),

    ("right wrist", "right thumb joint 1"),
    ("right thumb joint 1", "right thumb end"),

    ("right wrist", "right index joint 1"),
    ("right index joint 1", "right index end"),

    ("right wrist", "right middle joint 1"),
    ("right middle joint 1", "right middle end"),

    ("right wrist", "right ring joint 1"),
    ("right ring joint 1", "right ring end"),

    ("right wrist", "right little joint 1"),
    ("right little joint 1", "right little end"),
    #  left
    ("left shoulder", "left elbow"),
    ("left elbow", "left wrist"),

    ("left wrist", "left thumb joint 1"),
    ("left thumb joint 1", "left thumb end"),

    ("left wrist", "left index joint 1"),
    ("left index joint 1", "left index end"),

    ("left wrist", "left middle joint 1"),
    ("left middle joint 1", "left middle end"),

    ("left wrist", "left ring joint 1"),
    ("left ring joint 1", "left ring end"),

    ("left wrist", "left little joint 1"),
    ("left little joint 1", "left little end"),
)

reduced_name_neighbours = (
    ('head', 'jaw'),
    ('jaw', "right shoulder"),
    ("jaw", "left shoulder"),
    #  right
    ("right shoulder", "right elbow"),
    ("right elbow", "right wrist"),
    #  left
    ("left shoulder", "left elbow"),
    ("left elbow", "left wrist"),
)


def get_neighbours(joints=openpose_joints, body_type="SMPLH", num_node=None):
    key = [joints[point] for point in POINTS[body_type]]
    assert len(key) == num_node or num_node is None, f"Body type with {len(key)} nodes specified but {num_node}"
    names = reduced_name_neighbours if body_type == "SMPL" else name_neighbours

    neighbours = [(key.index(joints[beg]), key.index(joints[end])) for beg, end in names]
    neighbours += [(end, beg) for beg, end in neighbours]
    return neighbours


ears_group = (
    "right ear",
    "left ear",
)
head_group = (
    "head",
    "jaw",) + ears_group

upper_body_group = (
    "right wrist",
    "left wrist",
    "right elbow",
    "left elbow",
    "right shoulder",
    "left shoulder",
)
left_hand_group = (
    "left thumb end",
    "left index end",
    "left middle end",
    "left ring end",
    "left little end",

    #"left little knuckle",
    "left little joint 1",
    #"left little joint 2",
    #"left ring knuckle",
    "left ring joint 1",
    #"left ring joint 2",
    #"left middle knuckle",
    "left middle joint 1",
    #"left middle joint 2",
    #"left index knuckle",
    "left index joint 1",
    #"left index joint 2",
    #"left thumb knuckle",
    "left thumb joint 1",
    #"left thumb joint 2",
)
right_hand_group = (
    "right thumb end",
    "right index end",
    "right middle end",
    "right ring end",
    "right little end",

    #"right little knuckle",
    "right little joint 1",
    #"right little joint 2",
    #"right ring knuckle",
    "right ring joint 1",
    #"right ring joint 2",
    #"right middle knuckle",
    "right middle joint 1",
    #"right middle joint 2",
    #"right index knuckle",
    "right index joint 1",
    #"right index joint 2",
    #"right thumb knuckle",
    "right thumb joint 1",
    #"right thumb joint 2",
)


def get_indexes(sub_points, joints=holistic_joints, body_type="smplh"):
    # key = [joints[point] for point in POINTS[body_type]]
    return [POINTS[body_type].index(sub_point) for sub_point in sub_points]
