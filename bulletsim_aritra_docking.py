import pybullet as p 
import numpy as np 
import matplotlib.pyplot as plt 
import pybullet_data
import time
import math
from PIL import Image, ImageDraw
import yaml
from dynamics import QuarterRoboatDynamics

import torch 
import os
import scipy.ndimage
abs_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load(open(f"{abs_path}/cfg_docking.yaml"))
# CONFIG = yaml.safe_load(open(f"{abs_path}/cfg_onrt.yaml"))
import logging 
sim_logger = logging.getLogger("sim")
sim_logger.setLevel(logging.INFO)

# Configure the logger
file_handler = logging.FileHandler("sim.log", mode="w")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
sim_logger.addHandler(file_handler)

sim_logger.info("Logging from sim file.")


GRID_SIZE = 1000         # 1000 x 1000 grid
RESOLUTION = 0.1         # Each grid cell represents 0.1m
SUB_GRID_SIZE = 80       # Sub-grid size 80 x 80 for an 8m x 8m region
GLOBAL_COST_MAP = False

class Agent:
    def __init__(self, device, agent_cfg):
        self.device = device
        self.initial_pose = agent_cfg['initial_pose']
        self.initial_goal = agent_cfg['initial_goal']
        self.height = 0.075
        self.state = torch.tensor([self.initial_pose[0], self.initial_pose[1], self.initial_pose[2], 0, 0, 0], device=self.device)
        self.pos = torch.tensor([self.state[0], self.state[1], 0.075], device=self.device)
        self.rot = torch.tensor([np.pi, 0, self.state[2]], device=self.device)
        self.lin_vel = torch.tensor([self.state[3], self.state[4], 0], device=self.device)
        self.ang_vel = torch.tensor([0, 0, self.state[5]], device=self.device)
        
        # self.startOrientation = p.getQuaternionFromEuler([np.pi, 0, 0])
# "E:\\interaction_aware_mppi\\examples\\roboats\\aritra.urdf"

        # self.robotId = p.loadURDF("aritra.urdf", basePosition=self.pos, baseOrientation=p.getQuaternionFromEuler(self.rot), useFixedBase=True)
        urdf_path = os.path.abspath("aritra.urdf")
        try:
            self.robotId = p.loadURDF(urdf_path, basePosition=self.pos, baseOrientation=p.getQuaternionFromEuler(self.rot), useFixedBase=True)
        except Exception as e:
            print(f"Error loading URDF: {e}")

        self.urdf_id = self.robotId
        
        p.setCollisionFilterGroupMask(self.urdf_id, -1, 0, 0, physicsClientId=0)

        goal_urdf_path = 'urdf_files/sphere.urdf'
        self.goal_id = p.loadURDF(goal_urdf_path, basePosition=[self.initial_goal[0], self.initial_goal[1], 0], useFixedBase=True)
        p.setCollisionFilterGroupMask(self.goal_id, -1, 0, 0, physicsClientId=0)
    

        self.z = 0.0
        self.startPos = [0, 0, self.z]  # Slightly above the water
        self.startOrientation = p.getQuaternionFromEuler([np.pi, 0, 0])
   
        # cid = p.createConstraint(self.robotId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, self.z])
        # p.changeConstraint(cid, maxForce=500)
        
        # cid = p.createConstraint(self.robotId, -1, -1, -1, p.JOINT_PLANAR, [0, 0, 1], [0, 0, 0], [0, 0, self.z])
        # p.changeConstraint(cid, maxForce=500)

        # CONFIG = yaml.safe_load(open(f"cfg_roboats.yaml"))
    #     self.dynamics = QuarterRoboatDynamics(
    #      cfg=CONFIG
    # )
    #     self.dynamics = VesselDynamics(
    #      cfg=CONFIG
    # )
        
        self.lidar_link_index = self.get_link_index(self.robotId, "lidar_link")
        print(f"lidarlink index{self.lidar_link_index}")
        if self.lidar_link_index is None:
            print("Error: LiDAR link not found in URDF")
            p.disconnect()
            exit()

        self.calculate_and_print_dimensions()


    def get_state(self):
        return self.state

    def update_state(self, urdf_id, state):
        self.state = state
        self.pos = torch.tensor([state[0], state[1], 0.075], device=self.device)
        self.rot = torch.tensor([np.pi, 0, state[2]], device=self.device)
        self.lin_vel = torch.tensor([state[3], state[4], 0], device=self.device)
        self.ang_vel = torch.tensor([0, 0, state[5]], device=self.device)
        p.resetBasePositionAndOrientation(self.urdf_id, self.pos.cpu().numpy(), p.getQuaternionFromEuler(self.rot.cpu().numpy()))
        p.resetBaseVelocity(self.urdf_id, linearVelocity=self.lin_vel.cpu().numpy(), angularVelocity=self.ang_vel.cpu().numpy())
        return

    def update_goal_position(self, new_goal_position):
    # new_goal_position should be a list or tuple like [x, y, z]
        p.resetBasePositionAndOrientation(self.goal_id, new_goal_position, [0, 0, 0, 1])  # [0, 0, 0, 1] is the quaternion for no rotation
    

    def calculate_and_print_dimensions(self):
        aabb = p.getAABB(self.robotId)
        min_coords, max_coords = aabb

        length = max_coords[0] - min_coords[0]
        width = max_coords[1] - min_coords[1]
        height = max_coords[2] - min_coords[2]

        print(f"Robot dimensions:")
        print(f"Length: {length:.2f} meters")
        print(f"Width: {width:.2f} meters")
        print(f"Height: {height:.2f} meters")

    def get_link_index(self, robot_id, link_name):
        for i in range(p.getNumJoints(robot_id)):
            joint_info = p.getJointInfo(robot_id, i)
            if joint_info[12].decode('utf-8') == link_name:
                return i
        return None

    def get_lidar_data(self, enable_noise=False, noise_std=0.1):
        # Get the state of the LiDAR link
        self.lidar_state = p.getLinkState(self.robotId, self.lidar_link_index)
        self.lidar_position = self.lidar_state[0]  # World position of the LiDAR
        self.lidar_orientation = self.lidar_state[1]  # World orientation of the LiDAR

        num_rays = 3600  # Number of rays for a full 360-degree scan
        ray_length = 50  # Maximum length of each ray

        ray_from = [self.lidar_position for _ in range(num_rays)]
        ray_to = []

        # Convert quaternion to Euler angles
        euler_orientation = p.getEulerFromQuaternion(self.lidar_orientation)
        yaw = euler_orientation[2]  # We only need yaw for 2D LiDAR

        for i in range(num_rays):
            angle = 2 * math.pi * i / num_rays + yaw  # Add yaw to get world frame angle
            ray_to.append([
                self.lidar_position[0] + ray_length * math.cos(angle),
                self.lidar_position[1] + ray_length * math.sin(angle),
                self.lidar_position[2]  # Keep the same Z as the LiDAR
            ])

        results = p.rayTestBatch(ray_from, ray_to)

        self.coordinates = [result[3] for result in results]
        self.distances = [result[2] * ray_length for result in results]

        if enable_noise:
            # Add Gaussian noise to the distances
            noise = np.random.normal(0, noise_std, len(self.distances))
            self.distances = [max(0, d + n) for d, n in zip(self.distances, noise)]  # Ensure non-negative distances

        return self.distances, self.lidar_position, self.coordinates
    
    def visualize_lidar_data(self):
        max_distance = 50  # Maximum distance for normalization
        num_points = len(self.distances)
        
        for i, distance in enumerate(self.distances):
            if distance < max_distance:
                angle = 2 * math.pi * i / num_points
                x = self.lidar_position[0] + distance * math.cos(angle)
                y = self.lidar_position[1] + distance * math.sin(angle)
                z = self.lidar_position[2]
                color = [1 - distance/max_distance, distance/max_distance, 0]
                p.addUserDebugLine(self.lidar_position, [x, y, z], color, 1, 0.1)


class ObjectiveCost:
    def __init__(self,goals=None,device="cuda:0") :
        self.costmap = torch.zeros(80, 80)
        self.nav_goals = torch.tensor(goals, device=device).unsqueeze(0)
        
    def compute_running_cost(self, state: torch.Tensor):
        return self._goal_cost(state[:, :, 0:2])  + self._vel_cost(state) + self._heading_to_goal(state)
    

class BulletSimulator:
    def __init__(self, cfg):
        self.physicsClient = p.connect(p.GUI, options="--opengl3")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0) # Disable grid lines
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.planeId = p.loadURDF("plane.urdf")

        p.changeVisualShape(self.planeId, -1, rgbaColor=[0, 0.3, 0.7, 1])  # Change the color of the plane

        self.water_textures = self.create_water_texture()
        texture_id = p.loadTexture(self.water_textures)
        p.changeVisualShape(self.planeId, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id)
        p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=50, cameraPitch=-50, cameraTargetPosition=[0, 0, 0])
        
        # p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0])
        # self.agent = Agent()
        

        self.axis_lines = [None, None, None]

        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        
        self.trajectory_spheres = []
        self.sphere_radius = 0.05
        self.trajectory_color = [1, 0, 0, 1]  # Red color for trajectory
        self.marker_color = [0, 1, 0, 1]

        CONFIG = yaml.safe_load(open(f"cfg_docking.yaml"))
        self.agent_dynamics = QuarterRoboatDynamics(
         cfg=CONFIG
    )
    #     CONFIG_2 = yaml.safe_load(open(f"cfg_onrt.yaml"))
    #     self.agent_dynamics = VesselDynamics(
    #      cfg=CONFIG_2
    # )
        self._device = cfg["device"]
        self._dt = cfg["dt"]
        self._first_plot = True
        self._dynamics = self.agent_dynamics.step
        self._nx = cfg['mppi']['nx']

        self.grid_resolution = 0.1  # meters per cell
        self.world_size = (100, 100)  # Size of the world in meters (example: 100m x 100m)
        self.goal_position = cfg['agents']['agent0']['initial_goal'][0:2]  # Example global goal position
        self.step_counter = 0
        self.agent_position = cfg['agents']['agent0']['initial_pose'][0:2]

        self.dockenv()
        time.sleep(3)
        print(f'num of agents {cfg["agents"].items()}')
        self._agents = {name: Agent(cfg['device'], agent_cfg) for name, agent_cfg in cfg["agents"].items()}
        # time.sleep(3)

    def step(self, action) -> torch.Tensor:
        # Extract the actions from the dictionary and stack them
        action_tensor = torch.stack([a for a in action.values()])
        # action_tensor = torch.stack([30 for a in action.values()])

        state_tensor = torch.stack([agent.get_state() for agent in self._agents.values()])

        observation_tensor, action_tensor = self._dynamics(state_tensor.unsqueeze(0), action_tensor.unsqueeze(0))

        observation_dict = {}

        for i, agent_name in enumerate(self._agents.keys()):
            observation_dict[agent_name] = observation_tensor[:,i,:].squeeze()
            self._agents[agent_name].update_state(i, observation_tensor[:,i,:].squeeze())

        return observation_dict



    def check_collision(self):
        for dock_id in self.docking_bay_ids:
            contact_points = p.getContactPoints(self.agent.robotId, dock_id)
            if len(contact_points) > 0:
                return True
        return False



    def dockenv(self):


        dock_width = 4  # Adjust this based on the actual width of your dock bay
        dock_spacing = 0.1  # Small gap between dock bays
        dock_start_x = 10  # Starting X position of the first dock bay
        dock_y_offset = -5  # Half the distance between the two lines of docks

        # Create positions for the first line of dock bays
        dock_positions_line1 = [
            [dock_start_x, dock_y_offset + i * (dock_width + dock_spacing), 0] for i in range(1)
        ]

        dock_width = 4  # Adjust this based on the actual width of your dock bay
        dock_spacing = 0.1  # Small gap between dock bays
        dock_start_x = -10  # Starting X position of the first dock bay
        dock_y_offset = 5 # Half the distance between the two lines of docks


        # Create positions for the second line of dock bays (facing the first line)
        # dock_positions_line2 = [
        #     [dock_start_x, -dock_y_offset - i * (dock_width + dock_spacing), 0] for i in range(4)
        # ]

        # Combine both lines of dock positions
        dock_positions = dock_positions_line1 #+ dock_positions_line2

        # Load the dock bays
        self.docking_bay_ids = []
        for i, pos in enumerate(dock_positions):
            if i < 4:  # First line
                orientation = p.getQuaternionFromEuler([0, 0, 0])  # No rotation
            else:  # Second line
                orientation = p.getQuaternionFromEuler([0, 0, math.pi])  # Rotate 180 degrees around Z-axis
            
            dock_id = p.loadURDF("dockbay.urdf", pos, orientation, useFixedBase=True)
            self.docking_bay_ids.append(dock_id)
#################################

        grid_size = (int(self.world_size[0] / self.grid_resolution), 
                     int(self.world_size[1] / self.grid_resolution))
        
        goal_x, goal_y = self.cartesian_to_grid(self.goal_position[0], self.goal_position[1])



        
        def initialize_distance_grid(goal_x, goal_y):
            sim_logger.info(f"goal {goal_x} {goal_y}")
            distance_grid = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.float32, device=CONFIG["device"])
            
            # Calculate distance from each cell to the goal point
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    # Convert grid indices to Cartesian coordinates
                    cartesian_x, cartesian_y = self.grid_to_cartesian(i, j)
                    # sim_logger.info(f"cartx {cartesian_x} carty {cartesian_y}")
                    # Calculate Euclidean distance to the goal
                    distance = np.sqrt((cartesian_x - goal_x)**2 + (cartesian_y - goal_y)**2) 
                    distance_grid[i, j] = distance * 2
                    
            return distance_grid

        
        if GLOBAL_COST_MAP: self.global_cost_map = initialize_distance_grid(self.goal_position[0], self.goal_position[1]) 
        else:
            print("NO GC MAP GENERATED !!")
            self.global_cost_map =torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.float32) #np.full(grid_size, 1.0)  # Base cost for open cells

        sim_logger.info(self.global_cost_map)

        # Translate world coordinates to grid indices
        def world_to_grid(x, y):
            return (int((x + 50) / self.grid_resolution), int((y + 50) / self.grid_resolution))

        # Define obstacles in world coordinates and convert to grid indices

        # Add rectangular obstacles based on URDF specifications

        # Define wall dimensions relative to dock center position
        wall_definitions = [
            {"offset": (0, 1.75), "size": (4, 0.1)},  # left_wall
            {"offset": (0, -1.75), "size": (4, 0.1)}, # right_wall
            {"offset": (2, 0), "size": (0.1, 3.5)},  # back_wall
        ]

        # Iterate over each dock position and add walls to the cost map
        for dock_center in dock_positions:
            dock_x, dock_y, _ = dock_center

            for wall in wall_definitions:
                offset_x, offset_y = wall["offset"]
                size_x, size_y = wall["size"]

                # Calculate wall's center position in world coordinates
                wall_center_x = dock_x + offset_x
                wall_center_y = dock_y + offset_y

                # Convert obstacle boundaries to grid indices
                x_min, x_max = wall_center_x - size_x / 2, wall_center_x + size_x / 2
                y_min, y_max = wall_center_y - size_y / 2, wall_center_y + size_y / 2

                grid_x_min, grid_x_max = self.cartesian_to_grid(x_min, 0)[0], self.cartesian_to_grid(x_max, 0)[0]
                grid_y_min, grid_y_max = self.cartesian_to_grid(0, y_min)[1], self.cartesian_to_grid(0, y_max)[1]

                # Apply high negative cost to the obstacle region
                self.global_cost_map[grid_x_min:grid_x_max, grid_y_min:grid_y_max] =500.0

        # Inflate obstacles to ensure a safety buffer around them
        # inflated_obstacles = scipy.ndimage.binary_dilation(
        #     self.global_cost_map == -10.0, structure=np.ones((3, 3))
        # )
        # self.global_cost_map[inflated_obstacles] = -10.0

        # # Inflate obstacles
        inflated_obstacles = scipy.ndimage.binary_dilation(
            self.global_cost_map == 500.0, structure=np.ones((20, 15))
        )
        self.global_cost_map[inflated_obstacles] = 500.0

        non_goal_dock_x_min, non_goal_dock_x_max = self.cartesian_to_grid(10-3, 0)[0], self.cartesian_to_grid(10+2, 0)[0]
        non_goal_dock_y_min, non_goal_dock_y_max = self.cartesian_to_grid(0, 2.5-1.5)[1], self.cartesian_to_grid(0, 2.5+1)[1]
        # non_goal_dock_x, non_goal_dock_y = self.cartesian_to_grid(10-0.5,5-0.5)
        #****************
        # Check inflated non_dock position, could be reason vessel is not going inside. Global map and sim is different. 
        # Lateral inversion is there from sim to map. Weights are fine 
        '''        self._goal_weight = 1.0
                self._back_vel_weight = 0.08
                self._rot_vel_weight =  0.01
                self._lat_vel_weight = 0.01
                self._heading_to_goal_weight =  3.0
                self._within_goal_weight = 1.0
                self._max_speed = 0.3
                self._max_speed_weight = 5.0
                self.local_cost = None 
                self.curr_pos = None  '''

        self.global_cost_map[non_goal_dock_x_min:non_goal_dock_x_max, non_goal_dock_y_min:non_goal_dock_y_max] = 700.0
       #****************
    #    =-9-=         # inflated_dock = scipy.ndimage.binary_dilation(
        #     self.global_cost_map == 50.0, structure=np.ones((12, 12))
        # )
        # self.global_cost_map[inflated_dock] = 50.0

        # Generate gradient cost towards the goal
        
        # for i in range(grid_size[0]):
        #     for j in range(grid_size[1]):
        #         dist_to_goal = np.sqrt((i - goal_x) ** 2 + (j - goal_y) ** 2) * self.grid_resolution
        #         self.global_cost_map[i, j] += dist_to_goal*2

        # Plot the global cost map after initialization
        plt.figure(figsize=(8, 8))
        plt.imshow(self.global_cost_map, cmap='hot', origin='lower', extent=(-40, 40, -40, 40))
        plt.colorbar(label="Cost")
        plt.title("Global Cost Map")
        plt.xlabel("X Position (Global Grid)")
        plt.ylabel("Y Position (Global Grid)")
        plt.savefig("grid_maps/global_cost_map.png")
        plt.close()

    def grid_to_cartesian(self, grid_x, grid_y):
        cartesian_x = (grid_x - GRID_SIZE / 2) * RESOLUTION
        cartesian_y = (grid_y - GRID_SIZE / 2) * RESOLUTION
        return cartesian_x, cartesian_y

    def cartesian_to_grid(self, cartesian_x, cartesian_y):
        grid_x = int(cartesian_x / RESOLUTION + GRID_SIZE / 2)
        grid_y = int(cartesian_y / RESOLUTION + GRID_SIZE / 2)
        grid_x = np.clip(grid_x, 0, GRID_SIZE - 1)
        grid_y = np.clip(grid_y, 0, GRID_SIZE - 1)
        return grid_x, grid_y

    def get_sub_grid(self, distance_grid, cartesian_x, cartesian_y):
        # Convert Cartesian to grid coordinates
        center_x, center_y = self.cartesian_to_grid(cartesian_x, cartesian_y)
        
        # Define the range for the sub-grid
        half_size = SUB_GRID_SIZE // 2
        start_x = max(center_x - half_size, 0)
        end_x = min(center_x + half_size, GRID_SIZE)
        start_y = max(center_y - half_size, 0)
        end_y = min(center_y + half_size, GRID_SIZE)
        
        # Extract the 80x80 sub-grid
        
        sub_grid = distance_grid[start_x:end_x, start_y:end_y]

        # max_value = self.global_cost_map.max()  # Find the maximum value in the sub-grid
        # if max_value > 0:  # Ensure the max value is not zero to avoid division by zero
        #     sub_grid = sub_grid / max_value
        
        # Ensure the sub-grid is exactly 80x80 by padding if necessary
        padded_sub_grid = torch.zeros((SUB_GRID_SIZE, SUB_GRID_SIZE), dtype=torch.float32)
        padded_sub_grid[:sub_grid.shape[0], :sub_grid.shape[1]] = sub_grid
        # self.plot_grids(distance_grid, padded_sub_grid, self.goal_position[0], self.goal_position[1], cartesian_x, cartesian_y)
        return padded_sub_grid

    def plot_grids(self, big_grid, sub_grid, goal_x, goal_y, center_x, center_y):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the full 1000x1000 grid with distance to the goal
        ax1.imshow(big_grid.numpy(), cmap='viridis', origin='lower')
        ax1.set_title("1000x1000 Distance Grid to Goal Point")
        ax1.set_xlabel("Grid X")
        ax1.set_ylabel("Grid Y")

        # Mark the goal and the center of the sub-grid on the big grid
        goal_grid_x, goal_grid_y = self.cartesian_to_grid(goal_x, goal_y)
        ax1.scatter(goal_grid_y, goal_grid_x, color='red', label="Goal Point")
        
        center_grid_x, center_grid_y = self.cartesian_to_grid(center_x, center_y)
        ax1.scatter(center_grid_y, center_grid_x, color='blue', label="Sub-grid Center")
        ax1.legend()

        # Plot the extracted 80x80 sub-grid
        ax2.imshow(sub_grid.numpy(), cmap='viridis', origin='lower')
        ax2.set_title("80x80 Sub-grid of Distances (8m x 8m Region)")
        ax2.set_xlabel("Sub-grid X")
        ax2.set_ylabel("Sub-grid Y")

        plt.show()


    def query_local_cost_map(self, agent_position):
        agent_x, agent_y = agent_position
        local_extent = 4  # Local frame extent is (-4, 4) in both x and y directions
        half_local_size = int(local_extent / self.grid_resolution)

        # Convert agent's world position to grid indices
        grid_x, grid_y = int((agent_x + 40) / self.grid_resolution), int((agent_y + 40) / self.grid_resolution)

        # Extract agent-centered local map with padding if near edges
        local_cost_map = self.global_cost_map[
            max(0, grid_x - half_local_size): min(grid_x + half_local_size, self.global_cost_map.shape[0]),
            max(0, grid_y - half_local_size): min(grid_y + half_local_size, self.global_cost_map.shape[1])
        ]

        # Pad to ensure 80x80 grid if near edges
        local_cost_map = np.pad(
            local_cost_map,
            ((max(0, half_local_size - grid_x), max(0, half_local_size - (self.global_cost_map.shape[0] - grid_x))),
             (max(0, half_local_size - grid_y), max(0, half_local_size - (self.global_cost_map.shape[1] - grid_y)))),
            mode='constant', constant_values=1.0
        )
        return local_cost_map


    def visualize_local_cost_map(self):
        # Increment step counter
        self.agent_position = (self._agents['agent0'].pos[0], self._agents['agent0'].pos[1])
        self.step_counter += 1
        local_cost_map = self.query_local_cost_map(self.agent_position)
        # local_cost_map = self.get_sub_grid(self.global_cost_map, self.agent_position[0], self.agent_position[1])
        # local_cost_map = [0]
        # Save plot every 100 steps
        # if self.step_counter % 20 == 0:
            
        #     plt.figure(figsize=(6, 6))
        #     plt.imshow(local_cost_map, cmap='hot', origin='lower')
        #     plt.colorbar(label="Cost")
        #     plt.title(f"Local Cost Map at Step {self.step_counter}")
        #     plt.xlabel("X Position (Local Grid)")
        #     plt.ylabel("Y Position (Local Grid)")

        #     # Save the plot with step count in the filename
        #     plt.savefig(f"grid_maps/local_cost_map_step_{self.step_counter}.png")
        #     plt.close()
    
        return local_cost_map


    def create_water_texture(self, size=512):
        base_color = (0, 100, 180, 255)  # Base water color (RGBA)
        wave_color = (100, 200, 255, 255)  # Wave color (RGBA)

        img = Image.new('RGBA', (size, size), base_color)
        draw = ImageDraw.Draw(img)

        # Draw some wavy lines to simulate water
        for i in range(0, size, 20):
            for x in range(size):
                y = i + int(math.sin(x / 50) * 10)
                draw.point((x, y), fill=wave_color)

        img.save("water_texture.png")
        return "water_texture.png"
    



    def run(self):
        try:
            start_time = time.time()
            frame = 0
            plot_data = []
            # plt.scatter(self.agent.dynamics.my_state[6], self.agent.my_state[7], c = 'r')
            current_pos, current_orn = p.getBasePositionAndOrientation(self.agent.robotId)
            print(f"init pos {current_pos} init orn {current_orn}")

            


            while True:
                current_time = time.time() - start_time

                # if int(current_time) <= 10:
                    # state = self.agent.dynamics.agent.step(delta_c = np.radians(0), n_c = 115.5 / 60)
                # else: 
                # state = self.agent.dynamics.agent.step(delta_c = np.radians(0), n_c = 115.5 / 60)


                # u, v, w = state[0], state[1], state[2]
                # roll_rate, pitch_rate, yaw_rate = state[3], state[4], state[5]
                # new_x = state[6]
                # new_y = state[7]
                # neworn = state[9:13]
                # newyaw = kin.quat_to_eul(neworn)[2]
                # neworn =  p.getQuaternionFromEuler([0, 0, newyaw])

                current_pos, current_orn = p.getBasePositionAndOrientation(self.agent.robotId)
                curryaw = p.getEulerFromQuaternion(current_orn)[2]
                currvel = p.getBaseVelocity(self.agent.robotId)
               
                curstatetens = torch.tensor([current_pos[0],current_pos[1], curryaw, currvel[0][0], currvel[0][1], currvel[1][2]], dtype=torch.float32)
                                        #   x, y, theta, vx, vy, omega ])
                actiontens = 5*torch.tensor([1,-1,0,0], dtype=torch.float32)  # 1, 1, 1, 1 is forward thrust
                curstatetens = curstatetens.view(1, 1, 6)   # 0,0,1,1 is sway SB
                actiontens = actiontens.view(1, 1, 4)  # 1su 2su 3sw CW 4sw ACW
                '''
                          1 ^       ^ 2  ----> 3
                            |       |
                            |       |
                    4 ---->     
                
                '''
                
                nextstate, actions = self.agent.dynamics.step(curstatetens, actiontens)
                
                nextstate = nextstate[0,0,:]
                print(f"next state {nextstate}")
                p.resetBasePositionAndOrientation(self.agent.robotId, [nextstate[0], nextstate[1], self.agent.z], p.getQuaternionFromEuler([np.pi, 0, nextstate[2]]))

                
                # print(f"curr pos {current_pos} curr orn {p.getEulerFromQuaternion(current_orn)}")
                
                # new_pos = [new_x, new_y, self.agent.z]
                # print(f"new pos {new_pos} neworn {p.getEulerFromQuaternion(neworn)}")
                # p.resetBasePositionAndOrientation(self.agent.robotId, new_pos, neworn)
                # p.resetBaseVelocity(self.agent.robotId, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
                
                # sphere_id = p.createVisualShape(p.GEOM_SPHERE, radius=self.sphere_radius, rgbaColor=self.trajectory_color)
                # body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_id, basePosition=[new_x, new_y, 1+self.agent.z])
                # self.trajectory_spheres.append(body_id)
                
                marker_id = p.createVisualShape(p.GEOM_SPHERE, radius=self.sphere_radius, rgbaColor=self.marker_color)
                body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker_id, basePosition=[0, 0, 0.7+self.agent.z])
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker_id, basePosition=[-1.5, 0, 0.7+self.agent.z])
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker_id, basePosition=[3, 0, 0.7+self.agent.z])
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker_id, basePosition=[0, 3, 0.7+self.agent.z])
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker_id, basePosition=[10, 0, 0.7+self.agent.z])

                # p.resetBasePositionAndOrientation(self.agent.robotId, new_pos, neworn)

                # Set the velocity of the robot
                # p.resetBaseVelocity(self.agent.robotId, 
                #                     linearVelocity=[u, v, 0],  # Set vertical velocity to 0
                #                     angularVelocity=[roll_rate, pitch_rate, yaw_rate])
                # print(f"vel {p.getBaseVelocity(self.agent.robotId)}")

                # p.addUserDebugLine(new_pos, [new_x+1, new_y, 1+self.agent.z], [1, 0, 0], 2, 0.1)  # X-axis (Red)
                # p.addUserDebugLine(new_pos, [new_x, new_y+1, 1+self.agent.z], [0, 1, 0], 2, 0.1)  # Y-axis (Green)
                # p.addUserDebugLine(new_pos, [new_x, new_y, self.agent.z+1], [0, 0, 1], 2, 0.1)  # Z-axis (Blue)

                
                # state_list = [x for x in state[6:8]]
                # plot_data.append(state_list)


                
                # lidar_data, lidar_position, cordinates = self.agent.get_lidar_data(enable_noise=True)
                # self.agent.visualize_lidar_data()
                if self.check_collision():
                    print("Collision detected! Exiting simulation.")
                    break
                p.stepSimulation()
                time.sleep(1./240.)
                # p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[new_x, new_y, 0])

                pos, orn = p.getBasePositionAndOrientation(self.agent.robotId)
                # p.resetBasePositionAndOrientation(self.agent.robotId, [pos[0], pos[1], self.agent.z], p.getQuaternionFromEuler([np.pi, 0, 0]))

                if abs(pos[2] - self.agent.z) > 0.01:
                    p.resetBasePositionAndOrientation(self.agent.robotId, [pos[0], pos[1], self.agent.z], p.getQuaternionFromEuler([np.pi, 0, 0]))
                # print(current_time)
                if int(current_time) >= 600:break

            # plot_data = np.array(plot_data).T
            # plt.scatter(plot_data[0], plot_data[1])
            # plt.savefig('dummy.png')
        except KeyboardInterrupt:
            p.disconnect()
            print("Simulation ended by user")
        finally:
            # Clean up: remove the temporary texture files

            import os
            for i in range(len(self.water_textures)):
                if os.path.exists(f"water_texture_{i}.png"):
                    os.remove(f"water_texture_{i}.png")


    def plot_trajectories(self, traj_dict):
        if self._first_plot:
            self.marker_ids = []
            colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1]]  # Add more colors if needed
            iterations = len(traj_dict)/len(self._agents)
            for i in range(int(iterations)):
                color = colors[i % len(colors)]  # Cycle through colors if there are more agents than colors
                for _ in range(len(self._agents)*traj_dict[next(iter(traj_dict))].size()[0]):
                    visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=color)
                    body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id)
                    self.marker_ids.append(body_id)
            self._first_plot = False

        for i, agent_name in enumerate(traj_dict.keys()):
            trajectory_positions = traj_dict[agent_name].cpu().numpy()
            for j, state in enumerate(trajectory_positions):
                position_3d = np.append(state[:2], 0)
                p.resetBasePositionAndOrientation(self.marker_ids[i*len(trajectory_positions) + j], position_3d, (0, 0, 0, 1))
        return



# simulator = BulletSimulator()
# simulator.run()
