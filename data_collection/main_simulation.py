# --- FIX COMPATIBILITÀ PYTHON 3.10+ ---
import collections
import collections.abc
import math
import fractions
import numpy as np
import os
import yaml
import cv2
import time
import logging

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
fractions.gcd = math.gcd
np.float = float 
# --------------------------------------

import pybullet as p
import pybulletX as px
import tacto

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def get_wrist_camera_image(robot_id, link_id=7):
    """Acquisisce l'immagine dalla telecamera sul polso (224x224 per OpenVLA)"""
    state = p.getLinkState(robot_id, link_id)
    pos, orn = state[4], state[5]
    rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    
    # Offset: 5cm avanti rispetto al link 7
    cam_pos = pos + rot_matrix.dot([0, 0, 0.05])
    target_pos = pos + rot_matrix.dot([0, 0, 0.2])
    up_vector = rot_matrix.dot([0, -1, 0])
    
    view_matrix = p.computeViewMatrix(cam_pos, target_pos, up_vector)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=5.0)
    
    _, _, rgb, _, _ = p.getCameraImage(width=224, height=224, 
                                        viewMatrix=view_matrix, 
                                        projectionMatrix=proj_matrix,
                                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)

def main():
    # 1. Configurazione Percorsi
    base_path = os.path.dirname(os.path.abspath(__file__))
    panda_urdf = os.path.join(base_path, "assets/franka_panda/panda.urdf")
    digit_yaml = os.path.join(base_path, "assets/digit/digit.yaml")
    
    # Caricamento manuale della configurazione TACTO dallo YAML
    with open(digit_yaml, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. Inizializzazione Mondo e Robot
    log.info("Initializing PyBulletX World")
    px.init() 
    
    # Caricamento Panda (fixed_base=True per pybulletX)
    panda = px.Body(urdf_path=panda_urdf)
    
    # 3. Posizione di Home (Radianti)
    home_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    for i in range(7):
        p.resetJointState(panda.id, i, home_positions[i])
    
    # Apri dita (ID 9 e 10)
    p.resetJointState(panda.id, 9, 0.04)
    p.resetJointState(panda.id, 10, 0.04)

    # 4. Inizializzazione TACTO
    # Passiamo la sezione 'tacto' dello yaml come keyword arguments
    digits = tacto.Sensor(**cfg['tacto'])
    
    # Aggiungi camere ai link 9 e 10 (fingers)
    digits.add_camera(panda.id, [9, 10])

    log.info("Setup completato. Avvio simulazione...")

    # Visualizzazione della telecamera utente
    p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, 
                                 cameraPitch=-30, cameraTargetPosition=[0.5, 0, 0.5])

    # Thread di simulazione per fisica fluida
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    try:
        while True:
            # A. Render TACTO (Finestre DIGIT)
            color_t, depth_t = digits.render()
            digits.updateGUI(color_t, depth_t)
            
            # B. Render Wrist Camera (OpenVLA Input)
            wrist_img = get_wrist_camera_image(panda.id)
            cv2.imshow("Wrist Camera View", wrist_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        t.stop()
        p.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
