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

base_path = os.path.join(
    os.getcwd(),
    "assets",
)

print("cwd =", os.getcwd())
print("exists =", os.path.exists("objects/sphere_small.urdf"))
p.setAdditionalSearchPath(base_path)

p.resetSimulation()
p.setGravity(0,0,-9.81)

robot_id = p.loadURDF(
    "franka_panda/panda_digit.urdf",
    [0,0,0],
    useFixedBase=True
)

sensor = tacto.Sensor(width=224, height=224)

sensor.add_camera(robot_id, [10])   # digit_left_link
sensor.add_camera(robot_id, [12])   # digit_right_link

# --------------------------------------------------
# print all links
# --------------------------------------------------

link_map = {}

print("\nLINKS\n")

for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)

    link_name = info[12].decode()

    link_map[link_name] = i

    print(i, link_name)

print("\nMAP =", link_map)

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
# cube
# --------------------------------------------------

# cube_collision = p.createCollisionShape(
#     p.GEOM_BOX,
#     halfExtents=[0.025]*3
# )

# cube_visual = p.createVisualShape(
#     p.GEOM_BOX,
#     halfExtents=[0.025]*3
# )

# cube_id = p.createMultiBody(
#     baseMass=0.1,
#     baseCollisionShapeIndex=cube_collision,
#     baseVisualShapeIndex=cube_visual,
#     basePosition=[0.55,0.0,0.3]
# )

# cube_id = p.loadURDF(
#     "objects/cube_small.urdf",
#     [0.55, 0.0, 0.3],
#     globalScaling=0.5
# )

# cube_body = TactoBody(
#     cube_id,
#     "local_cube.urdf",
#     global_scaling=0.05
# )

# sphere_id = p.loadURDF(
#     "objects/sphere_small.urdf",
#     [0.55, 0.0, 0.3],
#     globalScaling=0.5
# )

# sphere_body = TactoBody(
#     sphere_id,
#     "assets/objects/sphere_small.urdf",
#     global_scaling=0.5
# )

# cone_id = p.loadURDF(
#     "objects/cone.urdf",
#     [0.55, 0.0, 0.3],
#     globalScaling=0.5
# )

# cone_body = TactoBody(
#     cone_id,
#     "assets/objects/cone.urdf",
#     global_scaling=0.5
# )

# ball_id = p.loadURDF(
#     "objects/abstract_ball.urdf",
#     [0.55, 0.0, 0.3],
#     globalScaling=0.4
# )

# ball_body = TactoBody(
#     ball_id,
#     "assets/objects/abstract_ball.urdf",
#     global_scaling=0.4
# )

ball_id = p.loadURDF(
    "objects/sphere_small.urdf",
    [0.55, 0.0, 0.3],
    globalScaling=0.5
)

ball_body = TactoBody(
    ball_id,
    "assets/objects/sphere_small.urdf",
    global_scaling=0.5
)

sensor.add_body(ball_body)

home_pose = [
    0.085,
    -0.035,
    -0.085,
    -1.5,
    -0.000,
    1.885,
    0.785
]

print("Going to HOME pose...")

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


rp = [
    0.134,
    -0.042,
    -0.049,
    -1.894,
    -0.139,
    1.761,
    0.076
]

OPENING = 0.04

grasp_target = [0.55, 0.0, 0.31]

while True:
    joint_poses = p.calculateInverseKinematics(
        robot_id,
        ee_link,
        target,
        orientation,
        lowerLimits=ll,
        upperLimits=ul,
        jointRanges=jr,
        restPoses=rp
        # jointDamping=[0.2]*7
    )

    for j in range(7):
        p.setJointMotorControl2(
            robot_id,
            j,
            p.POSITION_CONTROL,
            targetPosition=joint_poses[j],
            force=200
        )



    p.stepSimulation()
    time.sleep(1/240)

    for _ in range(240):
        p.setJointMotorControl2(
            robot_id,
            9,
            p.POSITION_CONTROL,
            targetPosition=OPENING,
            force=5
        )

        p.setJointMotorControl2(
            robot_id,
            11,
            p.POSITION_CONTROL,
            targetPosition=OPENING,
            force=5
        )

        p.stepSimulation()
        time.sleep(1/240)

    for _ in range(400):

        joint_poses = p.calculateInverseKinematics(
            robot_id,
            ee_link,
            grasp_target,
            orientation,
            lowerLimits=ll,
            upperLimits=ul,
            jointRanges=jr,
            restPoses=rp
        )

        for j in range(7):
            p.setJointMotorControl2(
                robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[j],
                force=200
            )

        p.stepSimulation()
        time.sleep(1/240)

    for _ in range(300):

        p.setJointMotorControl2(
            robot_id,
            9,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=5
        )

        p.setJointMotorControl2(
            robot_id,
            11,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=5
        )

        p.stepSimulation()
        time.sleep(1/240)

    contacts = p.getContactPoints(robot_id, ball_id)

    print("contacts:", len(contacts))

    for c in contacts:
        print(
            "robot link:", c[3],
            "normal force:", c[9]
        )
        
    color, depth = sensor.render()
    print(len(color))

    while True:
        p.stepSimulation()

        sensor.update()
        color, depth = sensor.render()

        cv2.imshow("left_digit", color[0])
        cv2.imshow("right_digit", color[1])
        cv2.waitKey(1)

        # view = p.computeViewMatrix(
        #     cameraEyePosition=cam_pos,
        #     cameraTargetPosition=camera_target,
        #     cameraUpVector=up
        # )

        obj_pos, _ = p.getBasePositionAndOrientation(ball_id)

        hand_pos = np.array(state[4])
        hand_quat = state[5]

        R = np.array(
            p.getMatrixFromQuaternion(hand_quat)
        ).reshape(3,3)

        forward = R[:,0]
        up = -R[:,2]

        cam_pos = hand_pos + 0.8 * forward

        camera_target = hand_pos - 0.01 * forward

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
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb = np.reshape(rgba, (h,w,4))[:,:,:3]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow("external_camera", rgb)

    while True:

        for j in range(7):
            p.setJointMotorControl2(
                robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[j],
                force=200
            )

        p.setJointMotorControl2(
            robot_id,
            9,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=100
        )

        p.setJointMotorControl2(
            robot_id,
            11,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=100
        )

        p.stepSimulation()
        time.sleep(1/240)


# stabilize physics
for _ in range(500):
    p.stepSimulation()
    time.sleep(1/240)



# # open fingers
# p.resetJointState(robot_id, 8, 0.04)
# p.resetJointState(robot_id, 9, 0.04)

while True:
    p.stepSimulation()

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=cube_id
    )

    if len(contacts):
        print("CONTACTS:", len(contacts))

    for i in range(7):
        print(f"joint {i}: {joint_poses[i]:.3f}")

    num_joints = p.getNumJoints(robot_id)

    while True:
        p.stepSimulation()

        print("\nJOINT STATES:")

        for j in range(num_joints):
            state = p.getJointState(robot_id, j)
            pos = state[0]
            print(f"joint {j}: {pos:.3f}")

        time.sleep(1/240)

    contacts = p.getContactPoints(robot_id, table_id)

    for c in contacts:
        print(
            "link:",
            c[3],
            "normal force:",
            c[9]
        )

    time.sleep(1/240)