import pybullet as p
import pybullet_data
import os
import time
import math
import tacto
import numpy as np
import cv2

np.float = float
np.int = int
np.bool = bool

class TactoBody:
    def __init__(self, body_id, urdf_path, global_scaling=1.0):
        self.id = body_id
        self.urdf_path = urdf_path
        self.global_scaling = global_scaling

p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.resetSimulation()
p.setGravity(0,0,-9.81)

robot_id = p.loadURDF(
    "assets/franka_panda/panda_digit.urdf",
    [0,0,0],
    useFixedBase=True
)


# --------------------------------------------------
# table
# --------------------------------------------------

table_collision = p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=[0.3,0.4,0.025]
)

table_visual = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[0.3,0.4,0.025]
)

table_id = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=table_collision,
    baseVisualShapeIndex=table_visual,
    basePosition=[0.55,0.0,0.25]
)

# --------------------------------------------------
# objects
# --------------------------------------------------

OBJECTS = [
    {
        "name": "cage",
        "urdf": "assets/objects/cage.urdf",
        "scale": 0.5,
    },
    {
        "name": "decorative_ball",
        "urdf": "assets/objects/decorative_ball.urdf",
        "scale": 0.4,
    },
    {
        "name": "sphere_small",
        "urdf": "assets/objects/sphere_small.urdf",
        "scale": 0.5
    },
    {
        "name": "abstract_ball",
        "urdf": "assets/objects/abstract_ball.urdf",
        "scale": 0.4,
    },
    {
        "name": "disco_ball",
        "urdf": "assets/objects/disco_ball.urdf",
        "scale": 0.4,
    },
    {
        "name": "cube",
        "urdf": "assets/objects/cube_small.urdf",
        "scale": 0.5
    },
    {
        "name": "wood_dice",
        "urdf": "assets/objects/wood_dice.urdf",
        "scale": 0.5,
    },
    {
        "name": "greek_dice",
        "urdf": "assets/objects/greek_dice.urdf",
        "scale": 0.5,
    }
]

# --------------------------------------------------
# object definition
# --------------------------------------------------

def spawn_object(obj_cfg):

    x = 0.55 + np.random.uniform(-0.01, 0.01)
    y = np.random.uniform(-0.01, 0.01)

    # yaw = np.random.uniform(-np.pi, np.pi)
    yaw = 0

    obj_id = p.loadURDF(
        obj_cfg["urdf"],
        [x, y, 0.3],
        p.getQuaternionFromEuler([0, 0, yaw]),
        globalScaling=obj_cfg["scale"]
    )

    body = TactoBody(
        obj_id,
        obj_cfg["urdf"],
        global_scaling=obj_cfg["scale"]
    )

    return obj_id, body

# --------------------------------------------------
# collect external camera
# --------------------------------------------------

def capture_external_camera(
    robot_id,
    hand_link,
    object_id
):

    state = p.getLinkState(
        robot_id,
        hand_link,
        computeForwardKinematics=True
    )

    hand_pos = np.array(state[4])
    hand_quat = state[5]

    obj_pos, _ = p.getBasePositionAndOrientation(
        object_id
    )

    R = np.array(
        p.getMatrixFromQuaternion(hand_quat)
    ).reshape(3,3)

    forward = R[:,0]

    cam_pos = hand_pos + 0.8 * forward

    view = p.computeViewMatrix(
        cameraEyePosition=cam_pos,
        cameraTargetPosition=obj_pos,
        cameraUpVector=-R[:,2]
    )

    proj = p.computeProjectionMatrixFOV(
        fov=5,
        aspect=640/480,
        nearVal=0.01,
        farVal=2.0
    )

    w,h,rgba,depth,seg = p.getCameraImage(
        width=640,
        height=480,
        viewMatrix=view,
        projectionMatrix=proj,
        # renderer=p.ER_BULLET_HARDWARE_OPENGL
        renderer=p.ER_TINY_RENDERER
    )

    rgb = np.reshape(
        rgba,
        (h,w,4)
    )[:,:,:3]

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

# def capture_external_camera(robot_id, hand_link, object_id):

#     obj_pos, _ = p.getBasePositionAndOrientation(
#         object_id
#     )

#     cam_pos = [
#         obj_pos[0] + 0.25,
#         obj_pos[1] + 0.25,
#         obj_pos[2] + 0.15
#     ]

#     view = p.computeViewMatrix(
#         cameraEyePosition=cam_pos,
#         cameraTargetPosition=obj_pos,
#         cameraUpVector=[0,0,1]
#     )

#     proj = p.computeProjectionMatrixFOV(
#         fov=60,
#         aspect=640/480,
#         nearVal=0.01,
#         farVal=2.0
#     )

#     w,h,rgba,depth,seg = p.getCameraImage(
#         width=640,
#         height=480,
#         viewMatrix=view,
#         projectionMatrix=proj,
#         renderer=p.ER_TINY_RENDERER
#         # renderer=p.ER_BULLET_HARDWARE_OPENGL
#     )

#     rgb = np.reshape(rgba, (h,w,4))[:,:,:3]

#     return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# --------------------------------------------------
# save samples
# --------------------------------------------------

def save_sample(
    sample_dir,
    left_digit,
    right_digit,
    external_rgb,
    object_name,
    object_pose,
    joint_positions,
    contact_forces
):

    os.makedirs(sample_dir, exist_ok=True)

    cv2.imwrite(
        os.path.join(sample_dir, "digit_left.png"),
        left_digit
    )

    cv2.imwrite(
        os.path.join(sample_dir, "digit_right.png"),
        right_digit
    )

    cv2.imwrite(
        os.path.join(sample_dir, "external.png"),
        external_rgb
    )

    np.savez(
        os.path.join(sample_dir, "metadata.npz"),
        object_name=object_name,

        object_position=np.array(object_pose[0]),
        object_quaternion=np.array(object_pose[1]),

        joint_positions=np.array(joint_positions),

        contact_forces=np.array(contact_forces)
    )

# --------------------------------------------------
# home pose
# --------------------------------------------------

home_pose = [
    0.085,
    -0.035,
    -0.085,
    -1.5,
    -0.000,
    1.885,
    0.785
]

for _ in range(300):
    for j in range(7):
        p.setJointMotorControl2(
            robot_id,
            j,
            p.POSITION_CONTROL,
            targetPosition=home_pose[j],
            force=200
        )
    p.stepSimulation()
    time.sleep(1/240)

# --------------------------------------------------
# choose ee link
# --------------------------------------------------

link_map = {}

for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)

    link_name = info[12].decode()

    link_map[link_name] = i


ee_link = link_map["panda_grasptarget"]

hand_link = link_map["panda_hand"]

state = p.getLinkState(
    robot_id,
    hand_link,
    computeForwardKinematics=True
)

target = [0.55, 0.0, 0.5]

orientation = p.getQuaternionFromEuler([math.pi, 0, 0])


ll = [-2.9, -1.8, -2.9, -3.0, -2.9, -0.1, -2.9]
ul = [ 2.9,  1.8,  2.9,  0.0,  2.9,  3.7,  2.9]
jr = [u-l for l,u in zip(ll, ul)]
rp = [ 0.134, -0.042, -0.049, -1.894, -0.139, 1.761, 0.076]

grasp_target = [0.55, 0.0, 0.31]

# --------------------------------------------------
# main collection loop
# --------------------------------------------------


DATASET_ROOT = "dataset"
SAMPLES_PER_OBJECT = 10

os.makedirs(
    DATASET_ROOT,
    exist_ok=True
)

sample_counter = 0

for obj_cfg in OBJECTS:

    print(
        f"\nCollecting {obj_cfg['name']}"
    )

    object_folder = os.path.join(
        DATASET_ROOT,
        obj_cfg["name"]
    )

    os.makedirs(
        object_folder,
        exist_ok=True
    )

    for sample_idx in range(
        SAMPLES_PER_OBJECT
    ):

        print(
            f"sample {sample_idx+1}/10"
        )


        sensor = tacto.Sensor(width=224, height=224)

        sensor.add_camera(robot_id, [10])   # digit_left_link
        sensor.add_camera(robot_id, [12])   # digit_right_link


        for _ in range(300):
            for j in range(7):
                p.setJointMotorControl2(
                    robot_id,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=home_pose[j],
                    force=200
                )

            p.stepSimulation()
            time.sleep(1/240)

        obj_id, body = spawn_object(
            obj_cfg
        )


        print("Loaded:", obj_cfg["name"])
        print(p.getVisualShapeData(obj_id))


        aabb = p.getAABB(obj_id)

            
        # scale = 0.05 / max(
        #     aabb[1][0] - aabb[0][0],
        #     aabb[1][1] - aabb[0][1],
        #     aabb[1][2] - aabb[0][2],
        # )

        # p.resetBasePositionAndOrientation(
        #     obj_id,
        #     pos,
        #     orn
        # )
        print("AABB size:",
            np.array(aabb[1]) - np.array(aabb[0]))
        print("visual:", p.getVisualShapeData(obj_id)[0][3])


        sensor.add_body(body)

        obj_pos, _ = (
            p.getBasePositionAndOrientation(
                obj_id
            )
        )

        grasp_target = [
            obj_pos[0],
            obj_pos[1],
            0.31
        ]

        for _ in range(240):
            p.stepSimulation()
            time.sleep(1/240)

        # open gripper

        for _ in range(240):

            p.setJointMotorControl2(
                robot_id,
                9,
                p.POSITION_CONTROL,
                targetPosition=0.04,
                force=5
            )

            p.setJointMotorControl2(
                robot_id,
                11,
                p.POSITION_CONTROL,
                targetPosition=0.04,
                force=5
            )

            p.stepSimulation()

        # move to object

        for _ in range(400):

            joint_poses = (
                p.calculateInverseKinematics(
                    robot_id,
                    ee_link,
                    grasp_target,
                    orientation,
                    lowerLimits=ll,
                    upperLimits=ul,
                    jointRanges=jr,
                    restPoses=rp
                )
            )

            for j in range(7):

                p.setJointMotorControl2(
                    robot_id,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=
                    joint_poses[j],
                    force=200
                )

            p.stepSimulation()

        # close gripper

        for _ in range(300):

            p.setJointMotorControl2(
                robot_id,
                9,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=10
            )

            p.setJointMotorControl2(
                robot_id,
                11,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=10
            )

            p.stepSimulation()

        # settle contact

        for _ in range(50):

            p.stepSimulation()

        sensor.update()

        color, depth = sensor.render()

        left_digit = color[0]
        right_digit = color[1]

        external_rgb = capture_external_camera(
            robot_id,
            hand_link,
            obj_id
        )

        contacts = p.getContactPoints(
            robot_id,
            obj_id
        )

        forces = [
            [c[3], c[9]]
            for c in contacts
        ]

        joints = [
            p.getJointState(
                robot_id,
                j
            )[0]
            for j in range(7)
        ]

        pose = (
            p.getBasePositionAndOrientation(
                obj_id
            )
        )

        sample_dir = os.path.join(
            object_folder,
            f"sample_{sample_idx:03d}"
        )

        for _ in range(120):
            p.stepSimulation()
            time.sleep(1/240)

        save_sample(
            sample_dir,
            left_digit,
            right_digit,
            external_rgb,
            obj_cfg["name"],
            pose,
            joints,
            forces
        )

        print(
            f"saved -> {sample_dir}"
        )

        p.removeBody(obj_id)

        sample_counter += 1

print(
    f"\nDONE. Collected "
    f"{sample_counter} samples."
)
