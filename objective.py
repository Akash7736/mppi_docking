import torch
import time
import numpy as np
import onrt.module_kinematics as kin
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import yaml 
import logging 
main_logger = logging.getLogger("main")
main_logger.setLevel(logging.INFO)

# Configure the logger
file_handler = logging.FileHandler("objective.log", mode="w")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
main_logger.addHandler(file_handler)

main_logger.info("Logging from main file.")

GRID_SIZE = 1000         # 1000 x 1000 grid
RESOLUTION = 0.1         # Each grid cell represents 0.1m
SUB_GRID_SIZE = 80       # Sub-grid size 80 x 80 for an 8m x 8m region
abs_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load(open(f"{abs_path}/cfg_docking.yaml"))

class RoboatObjective(object):
    def __init__(self, goals, device="cuda:0", dock_estimator=None):
        self.device = device
        self.nav_goals = torch.tensor(goals, device=device).unsqueeze(0)
        self._goal_weight = 1.0
        self._back_vel_weight = 0.08
        self._rot_vel_weight =  0.01
        self._lat_vel_weight = 0.01
        self._heading_to_goal_weight =  3.0
        self._within_goal_weight = 1.0
        self._max_speed = 0.3
        self._max_speed_weight = 5.0


           # Dock-specific weights
        self._wall_clearance_weight = 4.0
        self._entrance_alignment_weight = 2.5
        self._dock_orientation_weight = 2.0
        self._safety_margin = 0.8
        
        # State tracking
        self.local_cost = None
        self.curr_pos = None
        self.dock_estimator = dock_estimator
        self.dock_state = None
        self.dock_uncertainty = None  
       

    def update_dock_state(self, dock_state, dock_covariance):
        """Update dock state and uncertainty information"""
        self.dock_state = dock_state
        self.dock_uncertainty = torch.diagonal(dock_covariance)[:3]  # Only position and orientation uncertainty

    def compute_running_cost(self, state: torch.Tensor):
        # print(f"STATE {state.shape}")
        print(f"AGENT POS {self.curr_pos}")
        lc_cost =  self._lcmap_cost(state)
        # print(f"LC cost {lc_cost}")
        gc = self._goal_cost(state[:, :, 0:2]) 
        vc = self._vel_cost(state)
        hc = self._heading_to_goal(state) 
        goc = self._goal_orientation_cost(state)
        # print(f"GC {gc} vc {vc} hc {hc} goc {goc}")
        # print(f"gc {gc}")
        total_cost = lc_cost + vc + hc + goc 
        # total_cost = gc+ vc + hc + goc 
        print(f"total cost {total_cost}")

        if self.dock_state is not None:
            total_cost += (
                self._dock_clearance_cost(state) +     # Wall clearance
                self._dock_alignment_cost(state) +     # Dock alignment
                self._entrance_cost(state)             # Entrance alignment
            )

        return  total_cost
        # return lc_cost
   

# class RoboatObjective(object):
#     def __init__(self, goals, device="cuda:0"):
#         self.nav_goals = torch.tensor(goals, device=device).unsqueeze(0)
#         print(f"NAV GOALs :{self.nav_goals.detach()}")
#         self._goal_weight = 2.0
#         self._back_vel_weight = 1.0
#         self._rot_vel_weight = 0.01
#         self._lat_vel_weight = 0.05
#         self._heading_to_goal_weight = 1.0
#         self._within_goal_weight = 8.0
#         self._max_speed = 0.3 # m/s
#         self._max_speed_weight = 5.0

#     def compute_running_cost(self, state: torch.Tensor):
#         print(f"STATE {state}")
#         return self._goal_cost(state[:, :, 0:2])  + self._vel_cost(state) + self._heading_to_goal(state) +  self._goal_orientation_cost(state)
   
    def _lcmap_cost(self, state: torch.Tensor):
        """Compute cost from local cost map"""
        if self.local_cost is None:
            return torch.zeros(state.shape[0], 1, device=self.device)

        cost = torch.zeros(state.shape[0], 1, device=self.device)
        for i in range(len(state)):
            x, y = state[i,0,0:2]
            curx, cury = self.curr_pos
            cost[i,0] = self.get_subgrid_value(x, y, self.local_cost, curx, cury)

        return cost


    def _dock_clearance_cost(self, state):
        """Compute cost based on clearance from dock walls"""
        if self.dock_state is None or self.dock_estimator is None:
            return torch.zeros_like(state[:, :, 0])

        # Get wall positions from dock state
        walls = self.dock_estimator.map_manager.dock_features['walls']
        if not walls:
            return torch.zeros_like(state[:, :, 0])

        # Compute minimum distance to walls
        vessel_positions = state[:, :, 0:2]
        min_distances = []
        
        for wall in walls:
            distances = self._point_to_line_distance(vessel_positions, wall)
            min_distances.append(distances)

        min_dist = torch.stack(min_distances).min(dim=0)[0]
        
        # Apply uncertainty weighting
        position_uncertainty = self.dock_uncertainty[:2].mean()
        uncertainty_factor = torch.exp(-position_uncertainty)
        
        return torch.exp(-min_dist/self._safety_margin) * self._wall_clearance_weight * uncertainty_factor

    def _dock_alignment_cost(self, state):
        """Compute cost based on alignment with dock"""
        if self.dock_state is None:
            return torch.zeros_like(state[:, :, 0])

        # Get current heading and dock orientation
        vessel_heading = state[:, :, 2]
        dock_orientation = self.dock_state[2]  # Assuming dock orientation is third component
        
        # Compute heading difference
        heading_diff = self._normalize_angle(vessel_heading - dock_orientation)
        
        # Weight by uncertainty
        orientation_uncertainty = self.dock_uncertainty[2]
        uncertainty_factor = torch.exp(-orientation_uncertainty)
        
        # Distance-based weighting
        distance = torch.norm(state[:, :, 0:2] - self.dock_state[:2], dim=2)
        distance_weight = torch.exp(-0.5 * distance)
        
        return (heading_diff ** 2) * self._dock_orientation_weight * uncertainty_factor * distance_weight

    def _entrance_cost(self, state):
        """Compute cost based on alignment with dock entrance"""
        if self.dock_state is None or 'entrance' not in self.dock_estimator.map_manager.dock_features:
            return torch.zeros_like(state[:, :, 0])

        entrance = self.dock_estimator.map_manager.dock_features['entrance']
        if not entrance:
            return torch.zeros_like(state[:, :, 0])

        # Compute desired entrance vector
        entrance_vec = torch.tensor(entrance[1]) - torch.tensor(entrance[0])
        desired_heading = torch.atan2(entrance_vec[1], entrance_vec[0])
        
        # Get vessel heading
        vessel_heading = state[:, :, 2]
        
        # Compute heading difference
        heading_diff = self._normalize_angle(vessel_heading - desired_heading)
        
        # Distance-based weighting
        entrance_center = (torch.tensor(entrance[0]) + torch.tensor(entrance[1])) / 2
        distance = torch.norm(state[:, :, 0:2] - entrance_center, dim=2)
        distance_weight = torch.exp(-0.5 * distance)
        
        return (heading_diff ** 2) * self._entrance_alignment_weight * distance_weight

    def _normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _point_to_line_distance(self, points, line):
        """Compute distance from points to line segment"""
        start, end = torch.tensor(line[0], device=self.device), torch.tensor(line[1], device=self.device)
        line_vec = end - start
        point_vec = points - start.unsqueeze(0).unsqueeze(0)
        
        # Compute projection
        line_length = torch.norm(line_vec)
        line_unit = line_vec / line_length
        projection = torch.sum(point_vec * line_unit, dim=2).clamp(0, line_length)
        
        # Compute closest point on line
        closest_point = start + line_unit * projection.unsqueeze(-1)
        
        return torch.norm(points - closest_point, dim=2)


    def get_subgrid_value(self, cartesian_x, cartesian_y, sub_grid, sub_grid_center_x, sub_grid_center_y):
        """
        Get the cell value in the sub-grid based on Cartesian coordinates.
        Returns high cost for out-of-bounds coordinates.
        """
        # Calculate the offset in meters from the sub-grid center
        offset_x = cartesian_x - sub_grid_center_x
        offset_y = cartesian_y - sub_grid_center_y

        # Convert the offset to grid indices within the 80x80 sub-grid
        grid_x = int(SUB_GRID_SIZE // 2 + offset_x / RESOLUTION)
        grid_y = int(SUB_GRID_SIZE // 2 + offset_y / RESOLUTION)

        # If coordinates are outside the sub-grid, return high cost
        if not (0 <= grid_x < SUB_GRID_SIZE and 0 <= grid_y < SUB_GRID_SIZE):
            return 1000.0  # High cost for out-of-bounds

        return sub_grid[grid_x, grid_y].item()

    def _goal_cost(self, positions):
        # print(f"POS:{positions}")
        # print(f"NAV GOALS:{self.nav_goals[:,:,:]}")
        gc = torch.linalg.norm(positions - self.nav_goals[:,:,:2], axis=2) * self._goal_weight
        # print(f"GOAL COST SHAPE {gc.shape}")
        return gc
    
    def _vel_cost(self, state):
        # convert velocities to body frame
        cos = torch.cos(state[:, :, 2])
        sin = torch.sin(state[:, :, 2])
        vel_body = torch.stack([state[:, :, 3] * cos + state[:, :, 4] * sin, -state[:, :, 3] * sin + state[:, :, 4] * cos], dim=2)
        
        # penalize velocities in the back, lateral and rotational directions
        back_vel_cost = torch.relu(-vel_body[:, :, 0]) * self._back_vel_weight
        lat_vel_cost = vel_body[:, :, 1] ** 2 * self._lat_vel_weight
        rot_vel_cost = state[:, :, 5] ** 2 * self._rot_vel_weight

        # Calculate the magnitude of the velocity
        vel_magnitude = torch.norm(vel_body, dim=2)

        # Penalize velocity magnitude exceeding max speed
        exceed_max_speed_cost = (vel_magnitude - self._max_speed) ** 2 * self._max_speed_weight

        return back_vel_cost + lat_vel_cost + rot_vel_cost + exceed_max_speed_cost
    
    def _heading_to_goal(self, state):
        # Get the heading of the agent
        theta = state[:, :, 2]
        # Get the vector pointing to the goal
        goal = self.nav_goals[:,:,:2] - state[:, :, 0:2]
        # Compute the angle between the heading and the goal
        angle = torch.atan2(goal[:, :, 1], goal[:, :, 0]) - theta
        # Normalize the angle to [-pi, pi]
        angle = (angle+ torch.pi) % (2 * torch.pi) - torch.pi
        cost = torch.where(torch.linalg.norm(state[:,:,0:2] - self.nav_goals[:,:,:2], axis=2)>0.5, angle**2, torch.zeros_like(angle))
        return cost * self._heading_to_goal_weight
    

    # def _heading_to_goal(self, state):
    #     # Get the current heading of the agent (theta)
    #     theta = state[:, :, 2]
        
    #     # Get the vector pointing to the goal
    #     goal = self.nav_goals[:, :, 0:2] - state[:, :, 0:2]
        
    #     # Compute the angle between the heading and the goal
    #     angle_to_goal = torch.atan2(goal[:, :, 1], goal[:, :, 0]) - theta
        
    #     # Normalize the angle to [-pi, pi]
    #     angle_to_goal = (angle_to_goal + torch.pi) % (2 * torch.pi) - torch.pi
        
    #     # Compute the distance to the goal
    #     distance_to_goal = torch.linalg.norm(state[:, :, 0:2] - self.nav_goals[:, :, 0:2], axis=2)
        
    #     # Condition: If within 3m radius, use desired heading cost
    #     desired_heading = self.nav_goals[:, :, 2]  # Desired heading at the goal
    #     heading_diff = desired_heading - theta
    #     heading_diff = (heading_diff + torch.pi) % (2 * torch.pi) - torch.pi
        
    #     # Apply cost: angle to goal outside 3m, desired heading within 3m
    #     cost = torch.where(distance_to_goal > 3.0, (angle_to_goal ** 2)* self._within_goal_weight, (heading_diff **2)* self._heading_to_goal_weight)
        
    #     return cost 


    def _goal_orientation_cost(self, state):
        # Get the current heading of the agent (theta)
        theta = state[:, :, 2]
        desired_heading = self.nav_goals[:, :, 2]
        heading_diff = desired_heading - theta
        
        # Normalize the heading difference to the range [-pi, pi]
        heading_diff = (heading_diff + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Penalize the squared deviation from the desired heading
        orientation_cost = heading_diff ** 2
        positions = state[:, :, 0:2]
        distance_to_goal = torch.linalg.norm(state[:, :, 0:2] - self.nav_goals[:, :, 0:2], axis=2)

        cost = torch.where(distance_to_goal < 0.5, (heading_diff **2)* self._within_goal_weight, torch.zeros_like(distance_to_goal)) 

        return cost



    def get_goals(self):
        return self.nav_goals.squeeze(0)
    
    def set_goals(self, goals):
        self.nav_goals = torch.tensor(goals, device=self.nav_goals.device).unsqueeze(0)
        return None



class ONRTObjective(object):
    def __init__(self, goals, device="cuda:0"):
        self.nav_goals = torch.tensor(goals, device=device).unsqueeze(0)
        # print(f"NAV GOALs :{self.nav_goals.detach()}")
        self._goal_weight = 2.0
        self._back_vel_weight = 1.0
        self._rot_vel_weight = 0.1
        self._lat_vel_weight = -5.0
        self._heading_to_goal_weight = 1.0
        self._within_goal_weight = 8.0
        self._max_speed = 0.3 # m/s
        self._max_speed_weight = 5.0

    def compute_running_cost(self, state: torch.Tensor):
        # print(f"STATE {state}")
        return self._goal_cost(state[:, :, 6:8])  + self._vel_cost(state) + self._heading_to_goal(state) +  self._goal_orientation_cost(state)
   
    def _goal_cost(self, positions):
        # print(f"POS:{positions}")
        # print(f"NAV GOALS:{self.nav_goals[:,:,:]}")
        return torch.linalg.norm(positions - self.nav_goals[:,:,:2], axis=2) * self._goal_weight
    
    def _vel_cost(self, state):
        # convert velocities to body frame
        vel_body = torch.stack([state[:, :, 0], state[:, :, 1]], dim=2)
        
        # penalize velocities in the back, lateral and rotational directions
        back_vel_cost = torch.relu(-vel_body[:, :, 0]) * self._back_vel_weight
        lat_vel_cost = vel_body[:, :, 1] ** 2 * self._lat_vel_weight
        rot_vel_cost = state[:, :, 5] ** 2 * self._rot_vel_weight

        # Calculate the magnitude of the velocity
        vel_magnitude = torch.norm(vel_body, dim=2)

        # Penalize velocity magnitude exceeding max speed
        exceed_max_speed_cost = (vel_magnitude - self._max_speed) ** 2 * self._max_speed_weight

        return back_vel_cost + lat_vel_cost + rot_vel_cost + exceed_max_speed_cost
    
    # In module_kinematics.py
    def quat_to_eul_torch(self,quat):
        """
        Convert quaternion to Euler angles with batched torch tensors
        quat: torch tensor of shape [batch_size, num_samples, 4]
        returns: torch tensor of shape [batch_size, num_samples, 3]
        """
        # Extract quaternion components
        qw = quat[:, :, 0]
        qx = quat[:, :, 1]
        qy = quat[:, :, 2]
        qz = quat[:, :, 3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * torch.tensor(np.pi/2, dtype=torch.float32, device=quat.device),
            torch.asin(sinp)
        )
        
        

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        # Stack the angles along the last dimension
        euler = torch.stack([roll, pitch, yaw], dim=2)
        
        return euler

    # In objective.py
    def _heading_to_goal(self, state):
        """
        Compute heading to goal cost
        state: tensor of shape [batch_size, num_samples, state_dim]
        """
        # Extract quaternion components
        quat = state[:, :, 9:13]  # Get quaternion part
        
        # Get the heading of the agent
        quatpose = self.quat_to_eul_torch(quat)
        theta = quatpose[:, :, 2]  # Extract yaw angle
        
        # Get the vector pointing to the goal
        goal = self.nav_goals[:, :, :2] - state[:, :, 6:8]
        
        # Compute the angle between the heading and the goal
        angle = torch.atan2(goal[:, :, 1], goal[:, :, 0]) - theta
        
        # Normalize the angle to [-pi, pi]
        angle = (angle + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Calculate distance to goal
        distance = torch.linalg.norm(state[:, :, 6:8] - self.nav_goals[:, :, :2], dim=2)
        
        # Apply cost only when distance > 0.5
        cost = torch.where(distance > 0.5, angle**2, torch.zeros_like(angle))
        
        return cost * self._heading_to_goal_weight

    def _goal_orientation_cost(self, state):
        """
        Compute goal orientation cost
        state: tensor of shape [batch_size, num_samples, state_dim]
        """
        # Extract quaternion components
        quat = state[:, :, 9:13]
        
        # Get the current heading of the agent
        quatpose = self.quat_to_eul_torch(quat)
        theta = quatpose[:, :, 2]  # Extract yaw angle
        
        # Get desired heading from navigation goals
        desired_heading = self.nav_goals[:, :, 2]
        
        # Calculate heading difference
        heading_diff = desired_heading - theta
        
        # Normalize the heading difference to [-pi, pi]
        heading_diff = (heading_diff + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Calculate distance to goal
        distance = torch.linalg.norm(state[:, :, 6:8] - self.nav_goals[:, :, :2], dim=2)
        
        # Apply cost only when distance < 0.5
        cost = torch.where(distance < 0.5, 
                        (heading_diff ** 2) * self._within_goal_weight, 
                        torch.zeros_like(distance))
        
        return cost

    def get_goals(self):
        return self.nav_goals.squeeze(0)
    
    def set_goals(self, goals):
        self.nav_goals = torch.tensor(goals, device=self.nav_goals.device).unsqueeze(0)
        return None






class SocialNavigationObjective(object):
    def __init__(self, device="cuda:0"):
        self._device = device
        self._min_dist = 1.0
        self._width = 0.45
        self._height = 0.9
        self._coll_weight = 100.0
        self._rule_cross_radius = 5.0
        self._rule_headon_radius = 2.0
        self._rule_angle = torch.pi/4.0
        self._rule_min_vel = 0.05
        self._headon_weight = 20.0
        self._crossing_weight = 20.0
        self._standon_weight = 5.0


    def compute_running_cost(self, agents_states, init_agent_state, t):
        return self._rule_cost(agents_states, init_agent_state, t) + self._dynamic_collision_cost(agents_states)
    
    def _dynamic_collision_cost(self, agents_states):
        # Compute collision cost for each pair of agents
        n = agents_states.shape[1]
        i, j = torch.triu_indices(n, n, 1)  # Get indices for upper triangle of matrix
        agent_i_states = torch.stack([agents_states[:,index,:] for index in i])
        agent_j_states = torch.stack([agents_states[:,index,:] for index in j])

        # grid = self.create_occupancy_grid(agent_i_states)

        # Compute the distance between each pair of agents
        dist = torch.linalg.norm(agent_i_states[:, :, :2] - agent_j_states[:, :, :2], dim=2)
        # Compute the cost for each sample
        cost = torch.sum((dist < self._min_dist).float() * self._coll_weight, dim=0)

        return cost
    
    def create_occupancy_grid(self, agent_i_states):
        # Create a 1000x1000 pixel grid initialized with zeros
        grid = torch.zeros((1000, 1000))

        # Convert the agent's position from world coordinates to pixel coordinates
        # Assume that the world frame is centered at (500, 500) in pixel coordinates
        agent_position_pixel = (agent_i_states[:, :, :2] * 10 + 500).long()

        # Convert the agent's size from meters to pixels
        agent_size_pixel = (torch.tensor([self._height, self._width]) * 10).long()

        # Get the agent's heading
        theta = agent_i_states[:, :, 2]

        # Fill in the grid with ones where the agent is located
        for i in range(agent_position_pixel.shape[0]):
            x, y = agent_position_pixel[i]
            half_length, half_width = agent_size_pixel // 2

            # Create a rectangle representing the agent
            rectangle = torch.zeros((half_width * 2, half_length * 2))
            rectangle[half_width-half_width:half_width+half_width, half_length-half_length:half_length+half_length] = 1

            # Rotate the rectangle according to the agent's heading
            rotation_matrix = torch.tensor([[torch.cos(theta[i]), -torch.sin(theta[i])], [torch.sin(theta[i]), torch.cos(theta[i])]])
            rectangle = torch.einsum('ij,jkl->ikl', rotation_matrix, rectangle)

            # Add the rectangle to the grid
            grid[y-half_width:y+half_width, x-half_length:x+half_length] += rectangle

        return grid

    def _rule_cost(self, agents_states, init_agent_states, t):
        # Compute cost for head-on collisions
        n = agents_states.shape[1]
        a, b = torch.triu_indices(n, n, 1)  # Get indices for upper triangle of matrix
        i = torch.concat([a,b])
        j = torch.concat([b,a])
        agent_i_states = torch.stack([agents_states[:,index,:] for index in i])
        agent_j_states = torch.stack([agents_states[:,index,:] for index in j])
        init_agent_i_states = torch.stack([init_agent_states[:,index,:] for index in i])
        init_agent_j_states = torch.stack([init_agent_states[:,index,:] for index in j])

        # From here on, we assume the velocities are in the world frame
        # if not, rotate them before continuing!
        # Also, we assume EastNorthUp

        # test_i = torch.tensor([-0.5,0,0,0,1,0]).unsqueeze(0).unsqueeze(0)
        # test_j = torch.tensor([0.5,0,0,0,-1,0]).unsqueeze(0).unsqueeze(0)
        # self._check_right_side(test_i, test_j)
        # self._check_vel_headon(test_i, test_j)

        # compute useful stuff
        # Get the positions of the agents
        self.pos_i = agent_i_states[:, :, :2]
        self.pos_j = agent_j_states[:, :, :2]
        self.vel_i = agent_i_states[:, :, 3:5]
        self.vel_j = agent_j_states[:, :, 3:5]
        self.theta_i = agent_i_states[:, :, 2]
        self.init_pos_i = init_agent_i_states[:, :, :2]
        self.init_pos_j = init_agent_j_states[:, :, :2]
        self.init_vel_i = init_agent_i_states[:, :, 3:5]
        self.init_vel_j = init_agent_j_states[:, :, 3:5]

        # Compute the vector from the first agent to the second agent
        self.vij = self.pos_j - self.pos_i
        self.init_vij = self.init_pos_j - self.init_pos_i

        # Compute the angle between vij and vel_i
        self.angle_vij = torch.atan2(self.vij[:, :, 1], self.vij[:, :, 0])
        self.angle_vel_i = torch.atan2(self.vel_i[:, :, 1], self.vel_i[:, :, 0])
        self.angle_vel_j = torch.atan2(self.vel_j[:, :, 1], self.vel_j[:, :, 0])
        self.angle_vel_j_vel_i = self.angle_vel_j - self.angle_vel_i
        self.angle_vij_vel_i = self.angle_vij - self.angle_vel_i

        self.init_angle_vij = torch.atan2(self.init_vij[:, :, 1], self.init_vij[:, :, 0])
        self.init_angle_vel_i = torch.atan2(self.init_vel_i[:, :, 1], self.init_vel_i[:, :, 0])
        self.init_angle_vel_j = torch.atan2(self.init_vel_j[:, :, 1], self.init_vel_j[:, :, 0])
        self.init_angle_vel_j_vel_i = self.init_angle_vel_j - self.angle_vel_i
        self.init_angle_vij_vel_i = self.init_angle_vij - self.init_angle_vel_i

        self.magnitude_vij = torch.norm(self.vij, dim=2)
        self.magnitude_vel_i = torch.norm(self.vel_i, dim=2)
        self.magnitude_vel_j = torch.norm(self.vel_j, dim=2)

        self.init_magnitude_vij = torch.norm(self.init_vij, dim=2)
        self.init_magnitude_vel_i = torch.norm(self.init_vel_i, dim=2)
        self.init_magnitude_vel_j = torch.norm(self.init_vel_j, dim=2)

        right_side = self._check_right_side()
        headon = self._check_vel_headon()
        priority = self._check_priority(agent_i_states.shape[1])
        crossed_constvel = self._check_crossed_constvel(t)

        return torch.sum((right_side & headon) * self._headon_weight, dim=0) + torch.sum((priority & crossed_constvel) * self._crossing_weight, dim=0) + torch.sum((priority) * self._stand_on_cost(), dim=0)

    def _check_right_side(self):

        # # Compute the angle between vij and heading of agent i
        # angle_vij = torch.atan2(vij[:, :, 1], vij[:, :, 0])
        # angle = angle_vij - theta_i

        # compute angle diff and nomalize to [-pi, pi]
        # angle_diff = torch.atan2(torch.sin(self.angle_vij_vel_i + torch.pi/2), torch.cos(self.angle_vij_vel_i + torch.pi/2))

        angle_diff = (self.angle_vij_vel_i + torch.pi/2 + torch.pi) % (2 * torch.pi) - torch.pi

        # print(angle_diff - angle_diff2)

        # Check if the magnitude of vij is greater than the rule radius and the absolute difference between angle and pi/4 is less than the rule angle
        is_right_side = (self.magnitude_vij < self._rule_headon_radius) & (torch.abs(angle_diff) < self._rule_angle)

        return is_right_side
    
    def _check_vel_headon(self):
        # # Compute the angle between agents' headings
        # angle = theta_j - theta_i

        # compute angle diff and normalize to [-pi, pi]
        # angle_diff2 = torch.atan2(torch.sin(self.angle_vel_j_vel_i - torch.pi), torch.cos(self.angle_vel_j_vel_i - torch.pi))
        angle_diff = (self.angle_vel_j_vel_i - torch.pi + torch.pi) % (2 * torch.pi) - torch.pi

        # Check if the absolute difference between angle and pi is less than the rule angle
        is_headon = (torch.abs(angle_diff) < self._rule_angle) & (self.magnitude_vel_i >= self._rule_min_vel) & (self.magnitude_vel_j >= self._rule_min_vel)

        return is_headon
    
    def _check_priority(self, k):
        # # Compute the angle between vij and heading of agent i
        # angle_vij = torch.atan2(vij[:, :, 1], vij[:, :, 0])
        # angle = angle_vij - theta_i

        is_front_right = (self.init_magnitude_vij < self._rule_cross_radius) & (self.init_angle_vij_vel_i < 0) & (self.init_angle_vij_vel_i > -torch.pi/2)

        # # Compute the angle between agents' headings
        # angle_2 = theta_j - theta_i

        # compute angle diff and normalize to [-pi, pi]
        #angle_diff = torch.atan2(torch.sin(self.init_angle_vel_j_vel_i - torch.pi/2), torch.cos(self.init_angle_vel_j_vel_i - torch.pi/2))
        angle_diff = (self.init_angle_vel_j_vel_i - torch.pi/2 + torch.pi) % (2 * torch.pi) - torch.pi

        is_giveway_vel = (torch.abs(angle_diff) < self._rule_angle) & (self.init_magnitude_vel_i >= self._rule_min_vel) & (self.init_magnitude_vel_j >= self._rule_min_vel)

        return is_front_right.expand(-1, k) & is_giveway_vel.expand(-1, k)
    
    def _check_crossed_constvel(self, t):
        # pos_i = agent_i_states[:, :, :2]
        # init_pos_j = init_agent_j_states[:, :, :2]
        # vel_i = agent_i_states[:, :, 3:5]
        # init_vel_j = init_agent_j_states[:, :, 3:5]
        # theta_i = agent_i_states[:, :, 2]

        # find current position of agent j
        pos_j = self.init_pos_j + self.init_vel_j * t
        # make pos_j same size as pos_i
        pos_j = pos_j.expand(-1, self.pos_i.shape[1], -1)

        # Move pos_i n meters forward in the direction of heading
        n = 1.0  # replace with the actual distance you want to move
        dx = n * torch.cos(self.theta_i)
        dy = n * torch.sin(self.theta_i)

        pos_i_moved = self.pos_i.clone()
        pos_i_moved[:, :, 0] += dx
        pos_i_moved[:, :, 1] += dy

        # Compute the vector from the first agent to the second agent
        vij = pos_j - pos_i_moved

        # Compute the angle between vij and vel_i
        angle_vij = torch.atan2(vij[:, :, 1], vij[:, :, 0])
        angle = angle_vij - self.angle_vel_i

        # # Compute the angle between vij and heading of agent i
        # angle_vij = torch.atan2(vij[:, :, 1], vij[:, :, 0])
        # angle = angle_vij - theta_i

        # # Nomalize to [-pi, pi]
        # angle_diff = torch.atan2(torch.sin(angle + torch.pi/2), torch.cos(angle + torch.pi/2))
        # angle = torch.atan2(torch.sin(angle), torch.cos(angle))
        angle_diff = (angle + torch.pi/2 + torch.pi) % (2 * torch.pi) - torch.pi

        crossed_constvel = (angle_diff < 0) & (angle_diff > -torch.pi/2)
        # crossed_constvel = (angle < 0)

        return crossed_constvel
    
    def _stand_on_cost(self):
        # Compute the angle between vel_j and init_vel_j
        angle = self.angle_vel_j - self.init_angle_vel_j

        # compute angle diff and normalize to [-pi, pi]
        # angle_diff = torch.atan2(torch.sin(angle), torch.cos(angle))
        angle_diff = (angle + torch.pi) % (2 * torch.pi) - torch.pi

        return angle_diff ** 2 * self._standon_weight