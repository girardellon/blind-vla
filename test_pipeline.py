# ==============================================================================
# --- FIX COMPATIBILITÀ DI SISTEMA (DALLA REPO UFFICIALE TACTO) ---
# ==============================================================================
import collections
import collections.abc
import math
import fractions
import numpy as np

# Fix per Mapping / Iterable su Python 3.10+
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

# Fix per funzioni rimosse nelle nuove versioni di librerie matematiche
fractions.gcd = math.gcd
try:
    np.float = float
except AttributeError:
    pass
# ==============================================================================

import pybullet as p
import pybullet_data
import tacto
import h5py
import os

# --- WRAPPER PER TACTO ---
class TactoBody:
    """
    Involucro leggero per ingannare TACTO. 
    Fornisce gli attributi del file mesh necessari al renderer interno del sensore.
    """
    def __init__(self, body_id, urdf_path, global_scaling=1.0):
        self.id = body_id
        self.urdf_path = urdf_path
        self.global_scaling = global_scaling

def create_wrist_camera(robot_id, joint_id=7):
    """Calcola la matrice di vista per la telecamera sul polso (wrist camera)"""
    state = p.getLinkState(robot_id, joint_id)
    pos = state[0]
    orn = state[1]
    
    rot_matrix = p.getMatrixFromQuaternion(orn)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    
    forward_vector = rot_matrix[:, 2] # Direzione dello sguardo
    up_vector = rot_matrix[:, 1]      # Vettore "alto" della camera
    
    camera_pos = pos + forward_vector * 0.05 # leggero offset dal polso
    target_pos = camera_pos + forward_vector * 0.5
    
    view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=2.0)
    
    _, _, rgb, _, _ = p.getCameraImage(width=224, height=224, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
    return rgb[:, :, :3]

def run_pilot_episode():
    # 1. Avvia PyBullet in modalità GUI
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Carica il piano terrestre
    p.loadURDF("plane.urdf")
    
    # Carica il robot Franka Panda
    robotId = p.loadURDF("franka_panda/panda.urdf", [0.0, 0.0, 0.0], useFixedBase=True)
    
    # Identifica i link delle dita (9 e 10 nell'URDF standard)
    link_finger_sx = 9
    link_finger_dx = 10
    
    # Recuperiamo il percorso ASSOLUTO dell'URDF del cubo (indispensabile per il motore grafico di TACTO)
    cube_urdf_path = os.path.join(pybullet_data.getDataPath(), "cube.urdf")
    cube_scaling = 0.05
    cubeId = p.loadURDF(cube_urdf_path, [0.0, 0.0, 0.7], globalScaling=cube_scaling)
    
    print("✅ Scena PyBullet creata. Cubo e Robot istanziati.")

    # 2. Inizializzazione TACTO Standard (Senza config_path esterno)
    print("🔄 Inizializzazione modulo TACTO (caricamento default di sistema)...")
    bg = tacto.Sensor(width=224, height=224)
    
    # Registra le dita del robot come telecamere tattili (accetta tranquillamente gli ID nativi)
    bg.add_camera(robotId, [link_finger_sx, link_finger_dx])
    
    # Avvolgiamo il cubo nel wrapper in modo che TACTO possa leggerne il path mesh e lo scaling
    cube_body = TactoBody(cubeId, cube_urdf_path, global_scaling=cube_scaling)
    bg.add_body(cube_body)
    
    print("✅ TACTO agganciato ai link del robot e al cubo oggetto.")

    # 3. Step di simulazione per stringere l'oggetto
    p.setJointMotorControl2(robotId, 9, p.POSITION_CONTROL, targetPosition=0.01, force=20)
    p.setJointMotorControl2(robotId, 10, p.POSITION_CONTROL, targetPosition=0.01, force=20)
    
    print("🏋️ Chiusura gripper in corso...")
    for _ in range(20): 
        p.stepSimulation()
        # Nota: bg.update() rimosso perché deprecato; le pose si aggiornano automaticamente in bg.render()
        
    print("✅ Simulazione fisica completata.")

    # 4. Estrazione dei dati tramite API ufficiale .render()
    # .render() restituisce una lista di immagini (una per ogni camera registrata)
    color_images, depth_images = bg.render()
    print(f"DEBUG TACTO: Generati {len(color_images)} frame tattili dai sensori delle dita.")
    
    # Estrae il frame visivo reale dalla telecamera sul polso
    wrist_rgb = create_wrist_camera(robotId, joint_id=7)
    print(f"Shape Wrist Camera Image: {wrist_rgb.shape}")

    # 5. Scrittura del file HDF5 finale
    with h5py.File("test_sample_reale.h5", "w") as f:
        group = f.create_group("entry_000000")
        
        if len(color_images) > 0:
            # Prendiamo il frame del primo sensore (dito sinistro) e simuliamo la sequenza temporale
            finto_tactile_sequence = np.stack([color_images[0]] * 5, axis=0)
        else:
            finto_tactile_sequence = np.zeros((5, 224, 224, 3), dtype=np.uint8)
            
        group.create_dataset("tactile_frames", data=finto_tactile_sequence, dtype='uint8')
        group.create_dataset("visual_embedding", data=np.zeros((2176,), dtype=np.float16)) # Spazio allocato per OpenVLA
        group.create_dataset("force_vector", data=np.array([0.0, 0.0, 1.0], dtype=np.float32))
        group.create_dataset("object_id", data="primitive_cube")
        
    print("✅ Salvataggio su file 'test_sample_reale.h5' completato con successo!")
    p.disconnect()

if __name__ == "__main__":
    run_pilot_episode()
