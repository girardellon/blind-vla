import pybullet as p
import pybullet_data
import math
import time
import trimesh

def create_pure_digit_mesh():
    """Genera una mesh pulita per il DIGIT basata sulle dimensioni reali (20x20x30mm)"""
    # Creiamo un parallelepipedo centrato che rappresenta il guscio del DIGIT
    digit_mesh = trimesh.creation.box(extents=[0.02, 0.02, 0.03])
    digit_mesh.export("debug_digit.obj")
    
    # Crea un mini URDF minimale per l'ispezione
    urdf_content = """<?xml version="1.0" encoding="utf-8"?>
<robot name="digit_sensor">
  <link name="base_link">
    <contact>
      <lateral_friction value="1.0"/>
    </contact>
    <inertial>
      <mass value="0.02"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="debug_digit.obj" scale="1 1 1"/></geometry>
      <material name="blue"><color rgba="0.0 0.4 1.0 0.8"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="debug_digit.obj" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>"""
    
    with open("debug_digit.urdf", "w") as f:
        f.write(urdf_content)

def main():
    # 1. Inizializzazione della GUI
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, 0) # Nessuna gravità per non far cadere i pezzi disancorati
    
    # Carica il piano di riferimento
    p.loadURDF("plane.urdf", [0, 0, -0.5])
    
    # 2. Generazione e caricamento Asset
    create_pure_digit_mesh()
    
    # Carica Franka Panda nella posizione di origine
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    
    # Configura i giunti del braccio per portarlo in una posa flessa comoda davanti alla telecamera
    # Invece di lasciarlo verticale a 1.2m, lo pieghiamo per esporre il gripper al centro dello schermo
    target_joints = [0.0, 0.4, 0.0, -1.8, 0.0, 2.2, 0.8]
    for joint_idx, angle in enumerate(target_joints):
        p.resetJointState(robot_id, joint_idx, angle)
        
    # Indici dei link delle dita (gripper) nell'URDF nativo di PyBullet
    link_finger_sx = 9
    link_finger_dx = 10
    
    # Portiamo le dita a una semi-apertura stabile (0.02 metri ciascuna)
    p.resetJointState(robot_id, link_finger_sx, 0.02)
    p.resetJointState(robot_id, link_finger_dx, 0.02)
    
    print("\n🤖 [INFO] Franka Panda posizionato in posa di ispezione.")
    print("Mantenere la GUI aperta per analizzare l'allineamento delle dita.")

    # 3. Posizionamento statico dei DIGIT calcolato tramite le coordinate dei Link
    # Questo ciclo legge dove si trovano i link metallici in tempo reale e ci disegna sopra i DIGIT
    try:
        while p.isConnected():
            # Recupera lo stato globale (posizione e orientamento) del dito sinistro
            state_sx = p.getLinkState(robot_id, link_finger_sx)
            pos_sx, orn_sx = state_sx[0], state_sx[1]
            
            # Recupera lo stato globale del dito destro
            state_dx = p.getLinkState(robot_id, link_finger_dx)
            pos_dx, orn_dx = state_dx[0], state_dx[1]
            
            # --- AGGIUSTAMENTO OFFSET GEOMETRICO ---
            # Le dita di Panda hanno l'origine alla base del dito. Il sensore deve essere traslato
            # leggermente verso la punta (asse Z locale del link) e verso l'interno (asse X locale).
            # Per ora applichiamo un offset puramente visivo nello spazio globale per vedere dove cadono:
            offset_digit_sx = [pos_sx[0] + 0.015, pos_sx[1], pos_sx[2] + 0.04]
            offset_digit_dx = [pos_dx[0] - 0.015, pos_dx[1], pos_dx[2] + 0.04]
            
            # Disegnamo temporaneamente delle sfere di debug rosse nei punti esatti di ancoraggio delle dita
            p.addUserDebugParameter("--- Ispezione Visiva Attiva ---", 0, 0, 0)
            
            # Per mantenere l'ispezione interattiva senza appesantire la memoria, usiamo i comandi di rendering
            # Rinfresca la simulazione passiva
            p.stepSimulation()
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🏁 Ispezione interrotta dall'utente.")
    finally:
        p.disconnect()

if __name__ == "__main__":
    main()
