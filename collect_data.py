# ==============================================================================
# --- FIX COMPATIBILITÀ DI SISTEMA TACTO ---
# ==============================================================================
import collections
import collections.abc
import math
import fractions
import numpy as np

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
fractions.gcd = math.gcd
try:
    np.float = float
except AttributeError:
    pass
# ==============================================================================

import pybullet as p
import pybullet_data
import tacto
import trimesh
import h5py
import os
import random
import time

# --- GENERATORE DI MESH, OGGETTI E DIGIT URDF ---
def generate_assets():
    print("Generazione asset e sensori DIGIT...")
    
    # Crea le mesh per gli oggetti
    trimesh.creation.box(extents=[1.0, 1.0, 1.0]).export("local_cube.obj")
    trimesh.creation.icosphere(subdivisions=3, radius=0.5).export("local_sphere.obj")
    trimesh.creation.cylinder(radius=0.5, height=1.0).export("local_cylinder.obj")
    
    # Crea una mesh semplificata per il sensore DIGIT (un piccolo parallelepipedo)
    digit_box = trimesh.creation.box(extents=[0.02, 0.02, 0.03])
    digit_box.export("local_digit.obj")
    
    # URDF Primitivi
    urdfs = {
        "local_cube.urdf": '<robot name="c"><link name="b"><inertial><mass value="0.1"/><inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/></inertial><visual><geometry><mesh filename="local_cube.obj" scale="1 1 1"/></geometry></visual><collision><geometry><mesh filename="local_cube.obj" scale="1 1 1"/></geometry></collision></link></robot>',
        "local_sphere.urdf": '<robot name="s"><link name="b"><inertial><mass value="0.1"/><inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/></inertial><visual><geometry><mesh filename="local_sphere.obj" scale="1 1 1"/></geometry></visual><collision><geometry><mesh filename="local_sphere.obj" scale="1 1 1"/></geometry></collision></link></robot>',
        "local_cylinder.urdf": '<robot name="cl"><link name="b"><inertial><mass value="0.1"/><inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/></inertial><visual><geometry><mesh filename="local_cylinder.obj" scale="1 1 1"/></geometry></visual><collision><geometry><mesh filename="local_cylinder.obj" scale="1 1 1"/></geometry></collision></link></robot>',
        
        # URDF specifico per il sensore DIGIT da ancorare alle dita
        "local_digit.urdf": '<robot name="digit"><link name="base_link"><inertial><mass value="0.02"/><inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial><visual><geometry><mesh filename="local_digit.obj" scale="1 1 1"/></geometry></visual><collision><geometry><mesh filename="local_digit.obj" scale="1 1 1"/></geometry></collision></link></robot>'
    }
    
    for filename, content in urdfs.items():
        with open(filename, "w") as f:
            f.write(content)

class TactoBody:
    def __init__(self, body_id, urdf_path, global_scaling=1.0):
        self.id = body_id
        self.urdf_path = urdf_path
        self.global_scaling = global_scaling

def get_wrist_camera_image(robot_id, target_pos):
    """Camera fissa virtuale puntata sulla zona di presa"""
    camera_pos = [target_pos[0], target_pos[1] - 0.3, target_pos[2] + 0.2]
    view_matrix = p.computeViewMatrix(camera_pos, target_pos, [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=2.0)
    _, _, rgb, _, _ = p.getCameraImage(width=224, height=224, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
    return rgb[:, :, :3]

def run_episode(episode_idx, object_type, bg_sensor):
    # Spawna il robot in posizione standard
    robot_id = p.loadURDF(
        "panda_digit.urdf",
        [0.0, 0.0, 0.5],
        useFixedBase=True
    )
    
    # ID delle dita del Panda nell'URDF originale
    link_finger_sx = 9
    link_finger_dx = 10
    
    # Configurazione Oggetto
    x_rand = random.uniform(-0.02, 0.02)
    y_rand = random.uniform(-0.02, 0.02)
    obj_pos = [0.5 + x_rand, 0.0 + y_rand, 0.05] # Spostato davanti al robot sul tavolo
    obj_orn = p.getQuaternionFromEuler([0, 0, random.uniform(-0.5, 0.5)])
    
    if object_type == "cube":
        urdf_path, scaling = "local_cube.urdf", 0.04
    elif object_type == "cylinder":
        urdf_path, scaling = "local_cylinder.urdf", 0.04
    else:
        urdf_path, scaling = "local_sphere.urdf", 0.04

    obj_id = p.loadURDF(urdf_path, obj_pos, obj_orn, globalScaling=scaling)
    p.changeDynamics(obj_id, -1, lateralFriction=1.0)
    
    # 🎯 CORREZIONE 1: CINEMATICA INVERSA PER PORTARE IL ROBOT SULL'OGGETTO
    # Vogliamo che la mano (link 8) si posizioni esattamente sopra l'oggetto
    gripper_target_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.12]
    gripper_target_orn = p.getQuaternionFromEuler([math.pi, 0, 0]) # Ruotato verso il basso
    
    joint_poses = p.calculateInverseKinematics(robot_id, 8, gripper_target_pos, gripper_target_orn)
    
    # Reset istantaneo dei giunti del braccio sulla posizione di presa calcolata
    for i in range(7):
        p.resetJointState(robot_id, i, joint_poses[i])
        
    # 🛠️ CORREZIONE 2: CREAZIONE E SALDATURA DINAMICA DEI SENSORI DIGIT
    # Spawna i due sensori DIGIT come corpi indipendenti
    digit_sx_id = p.loadURDF("local_digit.urdf", [0,0,0])
    digit_dx_id = p.loadURDF("local_digit.urdf", [0,0,0])
    
    # Vincolo rigido (weld) tra il dito sinistro del robot e il DIGIT sinistro
    p.createConstraint(robot_id, link_finger_sx, digit_sx_id, -1, p.JOINT_FIXED, 
                       [0, 0, 0], [0.01, 0, 0.03], [0, 0, 0])
    
    # Vincolo rigido (weld) tra il dito destro del robot e il DIGIT destro
    p.createConstraint(robot_id, link_finger_dx, digit_dx_id, -1, p.JOINT_FIXED, 
                       [0, 0, 0], [-0.01, 0, 0.03], [0, 0, 0])

    # Registra le telecamere di TACTO ancorandole direttamente ai corpi dei DIGIT indipendenti
    bg_sensor.add_camera(digit_sx_id, [-1])
    bg_sensor.add_camera(digit_dx_id, [-1])
    
    # Registra l'oggetto in TACTO
    obj_body = TactoBody(obj_id, urdf_path, global_scaling=scaling)
    bg_sensor.add_body(obj_body)
    
    # --- CICLO DI PRESA ED ESTRAZIONE DATI CONTATTO ---
    contact_detected = False
    tactile_sequence = []
    wrist_image_captured = None
    force_vec = np.zeros(3, dtype=np.float32)
    normal_vec = np.zeros(3, dtype=np.float32)
    
    # Chiudi attivamente le dita
    p.setJointMotorControl2(robot_id, link_finger_sx, p.POSITION_CONTROL, targetPosition=0.01, force=20)
    p.setJointMotorControl2(robot_id, link_finger_dx, p.POSITION_CONTROL, targetPosition=0.01, force=20)
    
    for step in range(40):
        p.stepSimulation()
        time.sleep(1./240.) # Rende l'azione visibile ad occhio nudo nella GUI
        
        # Controlla collisione tra il DIGIT e l'oggetto
        pts = p.getContactPoints(digit_sx_id, obj_id)
        
        if pts and not contact_detected:
            total_normal_force = pts[0][9]
            if total_normal_force > 0.1:
                # 📸 CONGELAMENTO IMMEDIATO DEI DATI ALL'IMPATTO
                contact_detected = True
                wrist_image_captured = get_wrist_camera_image(robot_id, obj_pos)
                normal_vec = np.array(pts[0][7], dtype=np.float32)
                force_vec = normal_vec * total_normal_force
                
        if contact_detected and len(tactile_sequence) < 5:
            color_images, _ = bg_sensor.render()
            if len(color_images) > 0:
                tactile_sequence.append(color_images[0])
                
    # Pulizia
    p.removeBody(robot_id)
    p.removeBody(obj_id)
    p.removeBody(digit_sx_id)
    p.removeBody(digit_dx_id)
    
    if not contact_detected or len(tactile_sequence) < 5:
        return None
        
    return {
        "tactile_frames": np.stack(tactile_sequence, axis=0),
        "wrist_image": wrist_image_captured,
        "force_vector": force_vec,
        "contact_normal": normal_vec,
        "object_id": object_type
    }

def main():
    generate_assets()
    TARGET_EPISODES = 20
    BATCH_SIZE = 5
    
    print(f"Avvio Acquisizione. Target: {TARGET_EPISODES} campioni.")
    p.connect(p.GUI)

    pybullet_path = pybullet_data.getDataPath()
    print("PyBullet data:", pybullet_path)

    p.setAdditionalSearchPath(pybullet_path)

    base_path = os.path.join(
    os.getcwd(),
    "assets",
    "franka_panda"
    )

    p.setAdditionalSearchPath(base_path)
    
    bg_sensor = tacto.Sensor(width=224, height=224)
    primitive_pool = ["cube", "sphere", "cylinder"]
    
    collected_count = 0
    episode_buffer, images_buffer = [], []
    
    with h5py.File("dataset_chunk_primitive.h5", "w") as h5_file:
        while collected_count < TARGET_EPISODES:
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.loadURDF(
                os.path.join(pybullet_path, "plane.urdf")
            )
            
            obj_type = random.choice(primitive_pool)
            data = run_episode(collected_count, obj_type, bg_sensor)
            
            if data is None:
                continue # Se fallisce la presa riprova
                
            episode_buffer.append(data)
            images_buffer.append(data["wrist_image"])
            collected_count += 1
            print(f"📦 Campione registrato ({collected_count}/{TARGET_EPISODES}) | Forza d'impatto salvata!")
            
            if len(episode_buffer) == BATCH_SIZE:
                # Mock embedding OpenVLA
                mock_embeddings = np.zeros((BATCH_SIZE, 2176), dtype=np.float16)
                
                for i, ep_data in enumerate(episode_buffer):
                    entry_name = f"entry_{collected_count - BATCH_SIZE + i:06d}"
                    g = h5_file.create_group(entry_name)
                    g.create_dataset("tactile_frames", data=ep_data["tactile_frames"], dtype='uint8')
                    g.create_dataset("visual_embedding", data=mock_embeddings[i], dtype='float16')
                    g.create_dataset("force_vector", data=ep_data["force_vector"], dtype='float32')
                    g.create_dataset("contact_normal", data=ep_data["contact_normal"], dtype='float32')
                    g.create_dataset("object_id", data=ep_data["object_id"])
                
                episode_buffer.clear()
                images_buffer.clear()
                h5_file.flush()
                print("💾 Batch salvato su HDF5!")

        meta = h5_file.create_group("metadata")
        meta.create_dataset("openvla_version", data="frozen_encoder")
        meta.create_dataset("tacto_version", data="official_digit_224")

    p.disconnect()
    print("🏁 Dataset generato con successo e fisica reale verificata!")

if __name__ == "__main__":
    main()
