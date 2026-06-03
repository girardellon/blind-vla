import pybullet as p
import pybullet_data
import time

def create_procedural_urdf():
    print("🛠️ Generazione dell'URDF procedurale Robot + DIGIT...")
    
    # Creiamo un gripper completo usando solo primitive geometriche integrate (Box)
    # Questo elimina qualsiasi dipendenza da file .obj esterni o percorsi 'package://'
    urdf_content = """<?xml version="1.0" encoding="utf-8"?>
<robot name="procedural_gripper">
  
  <link name="panda_hand">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <geometry><box size="0.04 0.15 0.05"/></geometry>
      <material name="dark_grey"><color rgba="0.3 0.3 0.3 1.0"/></material>
    </visual>
    <collision>
      <geometry><box size="0.04 0.15 0.05"/></geometry>
    </collision>
  </link>

  <link name="panda_leftfinger">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><box size="0.015 0.01 0.05"/></geometry>
      <material name="light_grey"><color rgba="0.7 0.7 0.7 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><box size="0.015 0.01 0.05"/></geometry>
    </collision>
  </link>

  <link name="panda_rightfinger">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><box size="0.015 0.01 0.05"/></geometry>
      <material name="light_grey"><color rgba="0.7 0.7 0.7 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><box size="0.015 0.01 0.05"/></geometry>
    </collision>
  </link>

  <joint name="finger_joint_sx" type="prismatic">
    <parent link="panda_hand"/>
    <child link="panda_leftfinger"/>
    <origin xyz="0 0.04 0.025" rpy="0 0 0"/>
    <axis xyz="0 -1 0"/>
    <limit lower="0.0" upper="0.04" effort="20" velocity="0.2"/>
  </joint>

  <joint name="finger_joint_dx" type="prismatic">
    <parent link="panda_hand"/>
    <child link="panda_rightfinger"/>
    <origin xyz="0 -0.04 0.025" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0.0" upper="0.04" effort="20" velocity="0.2"/>
  </joint>

  <link name="digit_left">
    <visual>
      <geometry><box size="0.012 0.02 0.03"/></geometry>
      <material name="cyan"><color rgba="0.0 0.7 1.0 0.7"/></material>
    </visual>
    <collision>
      <geometry><box size="0.012 0.02 0.03"/></geometry>
    </collision>
  </link>

  <joint name="digit_left_fixed" type="fixed">
    <parent link="panda_leftfinger"/>
    <child link="digit_left"/>
    <origin xyz="0 -0.01 0.03" rpy="0 0 0"/>
  </joint>

  <link name="digit_right">
    <visual>
      <geometry><box size="0.012 0.02 0.03"/></geometry>
      <material name="cyan"><color rgba="0.0 0.7 1.0 0.7"/></material>
    </visual>
    <collision>
      <geometry><box size="0.012 0.02 0.03"/></geometry>
    </collision>
  </link>

  <joint name="digit_right_fixed" type="fixed">
    <parent link="panda_rightfinger"/>
    <child link="digit_right"/>
    <origin xyz="0 0.01 0.03" rpy="0 0 0"/>
  </joint>

</robot>
"""
    with open("procedural_gripper_digit.urdf", "w") as f:
        f.write(urdf_content)

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, 0)
    
    # Carica piano di terra standard
    p.loadURDF("plane.urdf", [0, 0, -0.05])
    
    # Genera l'URDF locale privo di dipendenze esterne
    create_procedural_urdf()
    
    # Carica l'end-effector assemblato
    gripper_id = p.loadURDF("procedural_gripper_digit.urdf", [0, 0, 0.1], useFixedBase=True)
    
    print("\n✅ Modello caricato con successo senza conflitti di mesh.")
    print("Elenco dei link logici registrati nell'assieme:")
    
    # Ottieni la mappatura dei link per TACTO
    num_joints = p.getNumJoints(gripper_id)
    for i in range(num_joints):
        joint_info = p.getJointInfo(gripper_id, i)
        joint_name = joint_info[1].decode('utf-8')
        link_name = joint_info[12].decode('utf-8')
        print(f"  └── Indice Giunto/Link: {i} | Nome Giunto: {joint_name:<16} | Nome Link Figlio: {link_name}")

    # Muove i motori per mostrare la cinematica simmetrica dei DIGIT (i blocchi azzurri)
    # I giunti 0 e 1 sono i prismatici delle dita
    p.setJointMotorControl2(gripper_id, 0, p.POSITION_CONTROL, targetPosition=0.01)
    p.setJointMotorControl2(gripper_id, 1, p.POSITION_CONTROL, targetPosition=0.01)

    print("\n🖥️ Ispezione visiva avviata nella GUI.")
    print("Controlla che i blocchi azzurri (DIGIT) siano solidali alle dita grigie.")
    
    try:
        while p.isConnected():
            p.stepSimulation()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n🏁 Ispezione terminata.")
    finally:
        p.disconnect()

if __name__ == "__main__":
    main()
