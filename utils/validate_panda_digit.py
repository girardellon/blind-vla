import pybullet as p
import pybullet_data
import os
import time

def main():
    # 1. Configurazione GUI
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 2. Setup path (punta alla cartella che contiene l'URDF)
    base_path = os.path.join(os.getcwd(), "assets", "franka_panda")
    p.setAdditionalSearchPath(base_path)
    
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    
    # 3. Caricamento Robot
    urdf_path = "panda_digit.urdf"
    try:
        robot_id = p.loadURDF(urdf_path, [0, 0, 0.5], useFixedBase=True)
        print(f"\n✅ Robot caricato correttamente da {urdf_path}")
    except Exception as e:
        print(f"\n❌ Errore critico nel caricamento: {e}")
        return
    
    # Table dimensions
    table_length = 0.6
    table_width  = 0.8
    table_height = 0.75

    table_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            table_length/2,
            table_width/2,
            0.025
        ]
    )

    table_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            table_length/2,
            table_width/2,
            0.025
        ]
    )

    table_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=table_collision,
        baseVisualShapeIndex=table_visual,
        basePosition=[
            0.55,    # in front of robot
            0.0,
            table_height
        ]
    )
 
    # Cube
    cube_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.025, 0.025, 0.025]
    )

    cube_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.025, 0.025, 0.025]
    )

    cube_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=cube_collision,
        baseVisualShapeIndex=cube_visual,
        basePosition=[0.55, 0.0, 0.8]
    )

    # Object mesh
    # obj_file = "local_cube.obj"

    # visual_shape = p.createVisualShape(
    #     p.GEOM_MESH,
    #     fileName=obj_file,
    #     meshScale=[1,1,1]
    # )

    # collision_shape = p.createCollisionShape(
    #     p.GEOM_MESH,
    #     fileName=obj_file,
    #     meshScale=[1,1,1]
    # )

    # object_id = p.createMultiBody(
    #     baseMass=0,
    #     baseCollisionShapeIndex=collision_shape,
    #     baseVisualShapeIndex=visual_shape,
    #     basePosition=[0.5, 0.0, 0.1]
    # )

    # p.setGravity(0, 0, 0)

    # 4. Debug: Verifica dove sono i sensori
    print("\n📋 Analisi posizioni Link (cerchiamo i DIGIT):")
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('utf-8')
        link_name = info[12].decode('utf-8')
        
        # Otteniamo la posizione globale del link
        ls = p.getLinkState(robot_id, i)
        pos = ls[0]
        
        if "digit" in link_name.lower():
            print(f"  >>> [TROVATO] Giunto {i}: {joint_name} | Link: {link_name} | Posizione: {pos}")
        else:
            print(f"  Link {i}: {link_name}")

    # 5. Muovi le dita per vedere se si muovono
    # (Gli indici 8 e 9 sono solitamente quelli delle dita nel Panda)
    p.setJointMotorControl2(robot_id, 8, p.POSITION_CONTROL, targetPosition=0.09)
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.09)

    # 6. Loop GUI
    try:
        while p.isConnected():
            p.stepSimulation()
            time.sleep(0.01)
    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    main()
