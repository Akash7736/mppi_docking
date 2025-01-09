
import torch 
import os
import yaml 
import numpy as np 

abs_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load(open(f"{abs_path}/cfg_docking.yaml"))

import torch
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

class DockMapManager:
    def __init__(self, device="cuda:0"):
        # Map parameters
        self.map_resolution = 0.1  # meters per cell
        self.local_map_size = 20   # 20x20 meters around dock
        self.grid_size = int(self.local_map_size / self.map_resolution)
        self.device = device
        
        # Initialize grid maps
        self.occupancy_grid = torch.zeros((self.grid_size, self.grid_size), device=device)
        self.confidence_map = torch.zeros_like(self.occupancy_grid)
        self.height_map = torch.zeros_like(self.occupancy_grid)
        
        # Feature management
        self.dock_features = {
            'walls': [],         # List of wall segments [(start_point, end_point), ...]
            'corners': [],       # List of corner points [(x, y), ...]
            'entrance_points': [],# List of entrance points [(x, y), ...]
            'reference_position': None,  # GPS reference point
            'last_update': None, # Timestamp of last update
        }
        
        # Update parameters
        self.decay_factor = 0.98    # Temporal decay for old observations
        self.min_confidence = 0.1
        self.max_confidence = 0.9
        self.log_odds_hit = 0.7     # Log odds for hit update
        self.log_odds_miss = -0.4   # Log odds for miss update
        self.max_range = 30.0       # Maximum reliable LiDAR range
        self.min_wall_points = 10   # Minimum points to fit a wall
        self.ransac_threshold = 0.1 # RANSAC threshold for wall fitting
        
        # ICP parameters
        self.icp_max_iterations = 50
        self.icp_tolerance = 1e-4

    def world_to_grid(self, points):
        """Convert world coordinates to grid indices"""
        x = ((points[:, 0] + self.local_map_size/2) / self.map_resolution).long()
        y = ((points[:, 1] + self.local_map_size/2) / self.map_resolution).long()
        return torch.stack([x, y], dim=1)

    def grid_to_world(self, indices):
        """Convert grid indices to world coordinates"""
        x = indices[:, 0] * self.map_resolution - self.local_map_size/2
        y = indices[:, 1] * self.map_resolution - self.local_map_size/2
        return torch.stack([x, y], dim=1)

    def transform_to_global(self, lidar_scan, vessel_pose):
        """Transform LiDAR points to global frame"""
        # Extract vessel position and orientation and ensure dtype consistency
        x, y, theta = vessel_pose.to(dtype=torch.float32)  # Convert to float32
        
        # Convert polar to cartesian coordinates (local frame)
        angles = torch.arange(0, 2*np.pi, lidar_scan['angle_increment'], 
                            device=self.device, dtype=torch.float32)
        ranges = torch.tensor(lidar_scan['ranges'], 
                            device=self.device, dtype=torch.float32)
        
        # Filter invalid measurements
        valid_mask = (ranges > 0.1) & (ranges < self.max_range)
        ranges = ranges[valid_mask]
        angles = angles[valid_mask]
        
        # Convert to cartesian (local frame)
        local_x = ranges * torch.cos(angles)
        local_y = ranges * torch.sin(angles)
        local_points = torch.stack([local_x, local_y], dim=1)
        
        # Create transformation matrix with consistent dtype
        R = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                        [torch.sin(theta), torch.cos(theta)]], 
                        device=self.device, dtype=torch.float32)
        t = torch.tensor([x, y], device=self.device, dtype=torch.float32)
        
        # Transform to global frame
        global_points = (R @ local_points.T).T + t
        
        return global_points

    def update_occupancy_grid(self, points, vessel_pose):
        """Update occupancy grid using ray tracing"""
        vessel_pos = vessel_pose[:2]
        grid_points = self.world_to_grid(points)
        vessel_grid = self.world_to_grid(vessel_pos.unsqueeze(0))[0]
        
        # Ray tracing for each point
        for end_point in grid_points:
            self.raytrace_update(vessel_grid, end_point)
            
        # Apply confidence updates
        valid_mask = (grid_points[:, 0] >= 0) & (grid_points[:, 0] < self.grid_size) & \
                    (grid_points[:, 1] >= 0) & (grid_points[:, 1] < self.grid_size)
        valid_points = grid_points[valid_mask]
        
        for point in valid_points:
            self.confidence_map[point[0], point[1]] = min(
                self.confidence_map[point[0], point[1]] + 0.1,
                self.max_confidence
            )

    def raytrace_update(self, start, end):
        """Bresenham's line algorithm for ray tracing"""
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # Update log odds
                if x == x1 and y == y1:
                    self.occupancy_grid[x, y] += self.log_odds_hit
                else:
                    self.occupancy_grid[x, y] += self.log_odds_miss
                    
                # Clip values
                self.occupancy_grid[x, y] = torch.clamp(self.occupancy_grid[x, y], -10, 10)
            
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

    def extract_dock_features(self, points):
        """Extract dock features using RANSAC"""
        # Convert to numpy for RANSAC
        points_np = points.cpu().numpy()
        
        # Find walls using RANSAC
        walls = self.ransac_line_fitting(points_np)
        
        # Find corners by intersecting walls
        corners = self.find_corners(walls)
        
        # Detect entrance points
        entrance_points = self.detect_entrance(walls, corners)
        
        return {
            'walls': walls,
            'corners': corners,
            'entrance_points': entrance_points
        }

    def ransac_line_fitting(self, points, max_iterations=100):
        """RANSAC for line fitting"""
        walls = []
        remaining_points = points.copy()
        
        while len(remaining_points) > self.min_wall_points:
            best_inliers = []
            best_model = None
            
            for _ in range(max_iterations):
                # Sample 2 random points
                sample_idx = np.random.choice(len(remaining_points), 2, replace=False)
                p1, p2 = remaining_points[sample_idx]
                
                # Fit line
                direction = p2 - p1
                normal = np.array([-direction[1], direction[0]])
                normal = normal / np.linalg.norm(normal)
                
                # Find inliers
                distances = np.abs(np.dot(remaining_points - p1, normal))
                inliers = remaining_points[distances < self.ransac_threshold]
                
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_model = (p1, p2)
            
            if len(best_inliers) > self.min_wall_points:
                # Refine wall using all inliers
                refined_wall = self.refine_wall(best_inliers)
                walls.append(refined_wall)
                
                # Remove inliers from remaining points
                mask = np.ones(len(remaining_points), dtype=bool)
                for inlier in best_inliers:
                    mask &= ~np.all(remaining_points == inlier, axis=1)
                remaining_points = remaining_points[mask]
            else:
                break
                
        return walls

    def refine_wall(self, points):
        """Refine wall segment using PCA"""
        centroid = np.mean(points, axis=0)
        _, v = np.linalg.eig(np.cov(points.T))
        direction = v[:, 0]
        
        # Project points onto line
        projected = np.dot(points - centroid, direction)
        min_proj = np.min(projected)
        max_proj = np.max(projected)
        
        # Get wall endpoints
        start = centroid + min_proj * direction
        end = centroid + max_proj * direction
        
        return (start, end)

    def find_corners(self, walls):
        """Find corners by intersecting walls"""
        corners = []
        
        for i, (start1, end1) in enumerate(walls):
            for start2, end2 in walls[i+1:]:
                # Find intersection
                A = np.vstack([end1 - start1, end2 - start2]).T
                if np.abs(np.linalg.det(A)) > 1e-6:  # Check if walls are not parallel
                    b = start2 - start1
                    x = np.linalg.solve(A, b)
                    if 0 <= x[0] <= 1 and 0 <= x[1] <= 1:  # Check if intersection is within segments
                        corner = start1 + x[0] * (end1 - start1)
                        corners.append(corner)
                        
        return corners

    def detect_entrance(self, walls, corners):
        """Detect dock entrance using wall configuration"""
        if len(corners) < 2:
            return []
            
        # Find parallel walls
        parallel_pairs = []
        for i, (start1, end1) in enumerate(walls):
            dir1 = end1 - start1
            dir1 = dir1 / np.linalg.norm(dir1)
            
            for j, (start2, end2) in enumerate(walls[i+1:], i+1):
                dir2 = end2 - start2
                dir2 = dir2 / np.linalg.norm(dir2)
                
                # Check if walls are parallel
                if np.abs(np.dot(dir1, dir2)) > 0.95:
                    dist = np.abs(np.cross(dir1, start2 - start1))
                    if 2.0 < dist < 5.0:  # Typical dock width
                        parallel_pairs.append((i, j))
                        
        # Find entrance points
        entrance_points = []
        for i, j in parallel_pairs:
            start1, end1 = walls[i]
            start2, end2 = walls[j]
            
            # Find closest endpoints
            endpoints = [(start1, start2), (start1, end2), (end1, start2), (end1, end2)]
            min_dist = float('inf')
            best_pair = None
            
            for p1, p2 in endpoints:
                dist = np.linalg.norm(p1 - p2)
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (p1, p2)
                    
            if best_pair is not None:
                entrance_points.extend(best_pair)
                
        return entrance_points

    def save_map(self, filename):
        """Save map and features to file"""
        map_data = {
            'occupancy_grid': self.occupancy_grid.cpu(),
            'confidence_map': self.confidence_map.cpu(),
            'height_map': self.height_map.cpu(),
            'dock_features': self.dock_features,
            'metadata': {
                'resolution': self.map_resolution,
                'size': self.local_map_size,
                'reference_position': self.dock_features['reference_position']
            }
        }
        torch.save(map_data, filename)

    def load_map(self, filename):
        """Load map and features from file"""
        map_data = torch.load(filename, map_location=self.device)
        self.occupancy_grid = map_data['occupancy_grid'].to(self.device)
        self.confidence_map = map_data['confidence_map'].to(self.device)
        self.height_map = map_data['height_map'].to(self.device)
        self.dock_features = map_data['dock_features']
        
        # Update parameters from metadata
        self.map_resolution = map_data['metadata']['resolution']
        self.local_map_size = map_data['metadata']['size']

    def visualize_map(self):
        """Visualize current map state"""
        plt.figure(figsize=(15, 5))
        
        # Plot occupancy grid
        plt.subplot(131)
        plt.imshow(self.occupancy_grid.cpu(), cmap='gray')
        plt.title('Occupancy Grid')
        plt.colorbar()
        
        # Plot confidence map
        plt.subplot(132)
        plt.imshow(self.confidence_map.cpu(), cmap='viridis')
        plt.title('Confidence Map')
        plt.colorbar()
        
        # Plot features
        plt.subplot(133)
        plt.imshow(self.occupancy_grid.cpu(), cmap='gray', alpha=0.5)
        
        # Plot walls
        for wall in self.dock_features['walls']:
            start, end = wall
            plt.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=2)
            
        # Plot corners
        corners = np.array(self.dock_features['corners'])
        if len(corners) > 0:
            plt.plot(corners[:, 0], corners[:, 1], 'go')
            
        # Plot entrance points
        entrance = np.array(self.dock_features['entrance_points'])
        if len(entrance) > 0:
            plt.plot(entrance[:, 0], entrance[:, 1], 'bo')
            
        plt.title('Features')
        plt.show()


    def get_local_cost_map(self, vessel_position):
        """
        Generate 80x80 local cost map around vessel position
        """
        LOCAL_SIZE = 80
        HALF_SIZE = LOCAL_SIZE // 2
        
        # Initialize empty cost map
        local_cost = torch.zeros((LOCAL_SIZE, LOCAL_SIZE), device=self.device)
        
        # Convert vessel position to grid coordinates
        vessel_grid_pos = self.world_to_grid(vessel_position.unsqueeze(0))[0]
        
        # Calculate grid boundaries
        start_x = max(0, int(vessel_grid_pos[0] - HALF_SIZE))
        end_x = min(self.grid_size, int(vessel_grid_pos[0] + HALF_SIZE))
        start_y = max(0, int(vessel_grid_pos[1] - HALF_SIZE))
        end_y = min(self.grid_size, int(vessel_grid_pos[1] + HALF_SIZE))
        
        # Calculate local map indices
        local_start_x = max(0, HALF_SIZE - vessel_grid_pos[0])
        local_start_y = max(0, HALF_SIZE - vessel_grid_pos[1])
        
        # Extract valid region from occupancy grid
        grid_region = self.occupancy_grid[start_x:end_x, start_y:end_y]
        conf_region = self.confidence_map[start_x:end_x, start_y:end_y]
        
        # Convert occupancy to cost
        region_cost = torch.where(grid_region > 0.5,
                                torch.tensor(500.0, device=self.device),
                                torch.tensor(0.0, device=self.device))
        
        # Apply confidence weighting
        region_cost = region_cost * conf_region
        
        # Place region in local cost map
        h, w = grid_region.shape
        local_cost[
            int(local_start_x):int(local_start_x + h),
            int(local_start_y):int(local_start_y + w)
        ] = region_cost
        
        return local_cost


    def _compute_feature_costs(self, vessel_position):
        """
        Compute costs based on dock features
        """
        feature_cost = torch.zeros_like(self.occupancy_grid)
        
        # Add costs for walls
        if self.dock_features['walls']:
            for wall in self.dock_features['walls']:
                start, end = wall
                wall_pts = torch.tensor([start, end], device=self.device)
                distances = self._point_to_line_distance(vessel_position.unsqueeze(0), wall_pts)
                feature_cost += torch.exp(-distances / 0.5)  # Safety margin of 0.5m
        
        # Add costs for entrance
        if self.dock_features['entrance_points']:
            entrance = self.dock_features['entrance_points']
            entrance_center = torch.tensor(
                [(entrance[0][0] + entrance[1][0])/2,
                (entrance[0][1] + entrance[1][1])/2],
                device=self.device
            )
            dist_to_entrance = torch.norm(vessel_position - entrance_center)
            feature_cost += torch.exp(-dist_to_entrance / 2.0)  # Attractive force to entrance
        
        return feature_cost

    def _point_to_line_distance(self, point, line):
        """
        Compute distance from point to line segment
        """
        start, end = line[0], line[1]
        line_vec = end - start
        point_vec = point - start
        
        # Compute projection
        line_length = torch.norm(line_vec)
        line_unit = line_vec / line_length
        projection = torch.sum(point_vec * line_unit, dim=-1).clamp(0, line_length)
        
        # Compute closest point on line
        closest = start + line_unit * projection.unsqueeze(-1)
        
        return torch.norm(point - closest, dim=-1)

import torch
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

class HybridDockEstimator:
    def __init__(self, device="cuda:0"):
        self.device = device
        
        # GPS parameters
        self.gps_std = 2.0  # GPS standard deviation in meters
        self.gps_update_rate = 1.0  # Hz
        self.last_gps_update = 0
        
        # LiDAR parameters
        self.lidar_max_range = 30.0
        self.lidar_min_range = 0.5
        self.angle_resolution = np.pi/180  # 1 degree
        
        # State estimation parameters
        self.state_dim = 7  # [x, y, theta, dock_x, dock_y, dock_theta, entrance_width]
        # State estimation parameters
        self.state = torch.zeros(self.state_dim, device=device, dtype=torch.float32)
        self.covariance = torch.eye(self.state_dim, device=device, dtype=torch.float32)
    
        # Noise parameters
        self.R_gps = torch.diag(torch.tensor([self.gps_std**2, self.gps_std**2], 
                                            device=device, dtype=torch.float32))
        self.R_lidar = torch.diag(torch.tensor([0.1**2, 0.1**2, 0.1**2], 
                                            device=device, dtype=torch.float32))
        self.Q = torch.eye(self.state_dim, device=device, dtype=torch.float32) * 0.1

        # Map manager
        self.map_manager = DockMapManager(device=device)
        
        # Feature matching parameters
        self.feature_match_threshold = 0.5
        self.max_feature_history = 10
        self.feature_history = []
        
        # Cost parameters
        self.distance_weight = 1.0
        self.orientation_weight = 2.0
        self.safety_weight = 3.0
        
        # Initialize Kalman filter parameters
        self._initialize_kalman()

    def _initialize_kalman(self):
        """Initialize Kalman filter matrices"""
        # State transition matrix
        self.F = torch.eye(self.state_dim, device=self.device)
        
        # Control input matrix
        self.B = torch.zeros((self.state_dim, 3), device=self.device)  # [v_x, v_y, omega]
        self.B[0, 0] = 1.0  # x velocity
        self.B[1, 1] = 1.0  # y velocity
        self.B[2, 2] = 1.0  # angular velocity

    def predict(self, control_input, dt):
        """
        Kalman filter prediction step
        control_input: [v_x, v_y, omega]
        """
        # Update state
        self.state = self.F @ self.state + self.B @ control_input * dt
        
        # Update covariance
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q * dt

    def update_gps(self, gps_reading, current_time):
        """
        Update state using GPS measurement
        """
        if current_time - self.last_gps_update < 1.0/self.gps_update_rate:
            return
            
        # GPS measurement matrix
        H_gps = torch.zeros((2, self.state_dim), device=self.device)
        H_gps[:2, :2] = torch.eye(2)  # GPS measures x, y position
        
        # Kalman update
        innovation = gps_reading - H_gps @ self.state
        S = H_gps @ self.covariance @ H_gps.T + self.R_gps
        K = self.covariance @ H_gps.T @ torch.inverse(S)
        
        self.state = self.state + K @ innovation
        self.covariance = (torch.eye(self.state_dim, device=self.device) - K @ H_gps) @ self.covariance
        
        self.last_gps_update = current_time

    def process_lidar(self, lidar_scan, vessel_pose):
        """
        Process LiDAR data and update map
        """
        # Transform LiDAR points to global frame
        global_points = self.map_manager.transform_to_global(lidar_scan, vessel_pose)
        
        # Update occupancy grid
        self.map_manager.update_occupancy_grid(global_points, vessel_pose)
        
        # Extract features
        features = self.map_manager.extract_dock_features(global_points)
        
        # Update feature history
        self.update_feature_history(features)
        
        return features

    def update_feature_history(self, features):
        """
        Maintain history of recent features
        """
        self.feature_history.append(features)
        if len(self.feature_history) > self.max_feature_history:
            self.feature_history.pop(0)

    def match_features(self, features):
        """
        Match new features with existing database
        """
        matched_features = {}
        
        for feature_type in ['walls', 'corners', 'entrance_points']:
            if feature_type in features and features[feature_type]:
                current_features = np.array(features[feature_type])
                if feature_type in self.map_manager.dock_features and self.map_manager.dock_features[feature_type]:
                    db_features = np.array(self.map_manager.dock_features[feature_type])
                    
                    # Use KD-tree for efficient nearest neighbor search
                    tree = KDTree(db_features)
                    distances, indices = tree.query(current_features)
                    
                    # Match features within threshold
                    matches = indices[distances < self.feature_match_threshold]
                    matched_features[feature_type] = matches
                    
        return matched_features

    def compute_measurement_model(self, features):
        """
        Compute measurement model from features
        """
        if not features['walls'] or not features['entrance_points']:
            return None
            
        # Extract measurements from features
        wall_angles = []
        wall_distances = []
        entrance_width = None
        
        # Process walls
        for start, end in features['walls']:
            direction = end - start
            angle = np.arctan2(direction[1], direction[0])
            distance = np.linalg.norm(start)
            wall_angles.append(angle)
            wall_distances.append(distance)
            
        # Process entrance points
        if len(features['entrance_points']) >= 2:
            p1, p2 = features['entrance_points'][:2]
            entrance_width = np.linalg.norm(p2 - p1)
            
        return torch.tensor([
            np.mean(wall_distances),
            np.mean(wall_angles),
            entrance_width if entrance_width is not None else 0.0
        ], device=self.device)

    def update_state(self, features):
        """
        Update state estimate using feature measurements
        """
        measurement = self.compute_measurement_model(features)
        if measurement is None:
            return
            
        # Measurement matrix
        H = torch.zeros((3, self.state_dim), device=self.device)
        H[0, 3:5] = torch.tensor([1.0, 1.0])  # dock position
        H[1, 5] = 1.0  # dock orientation
        H[2, 6] = 1.0  # entrance width
        
        # Kalman update
        innovation = measurement - H @ self.state
        S = H @ self.covariance @ H.T + self.R_lidar
        K = self.covariance @ H.T @ torch.inverse(S)
        
        self.state = self.state + K @ innovation
        self.covariance = (torch.eye(self.state_dim, device=self.device) - K @ H) @ self.covariance

    def compute_cost_map(self, vessel_state):
        """
        Generate 80x80 local cost map around vessel position
        """
        LOCAL_SIZE = 80
        HALF_SIZE = LOCAL_SIZE // 2
        
        # Initialize parameters - increased weights for better cost differentiation
        obstacle_weight = 1000.0  # High cost for obstacles
        safety_margin = 3  # Cells for safety margin (increased for wider safety zone)
        
        # Get vessel position
        vessel_position = vessel_state[:2]
        vessel_grid_pos = self.map_manager.world_to_grid(vessel_position.unsqueeze(0))[0]
        
        # Calculate grid boundaries
        start_x = max(0, int(vessel_grid_pos[0] - HALF_SIZE))
        end_x = min(self.map_manager.grid_size, int(vessel_grid_pos[0] + HALF_SIZE))
        start_y = max(0, int(vessel_grid_pos[1] - HALF_SIZE))
        end_y = min(self.map_manager.grid_size, int(vessel_grid_pos[1] + HALF_SIZE))
        
        # Initialize cost map
        local_cost = torch.zeros((LOCAL_SIZE, LOCAL_SIZE), device=self.device)
        
        # Extract occupancy and confidence regions
        grid_region = self.map_manager.occupancy_grid[start_x:end_x, start_y:end_y]
        conf_region = self.map_manager.confidence_map[start_x:end_x, start_y:end_y]
        
        # Create safety margin around obstacles
        obstacle_mask = (grid_region > 0.5).cpu().numpy()
        if obstacle_mask.any():
            from scipy.ndimage import distance_transform_edt
            distance_map = distance_transform_edt(~obstacle_mask)
            safety_cost = np.exp(-distance_map / safety_margin)
            safety_cost = torch.tensor(safety_cost, device=self.device)
        else:
            safety_cost = torch.zeros_like(grid_region)
        
        # Combine obstacle and safety costs
        obstacle_cost = torch.where(grid_region > 0.5,
                                torch.tensor(obstacle_weight, device=self.device),
                                torch.tensor(0.0, device=self.device))
        
        region_cost = (obstacle_cost + safety_cost * obstacle_weight/2) * conf_region
        
        # Place region in local cost map
        h, w = grid_region.shape
        local_start_x = max(0, HALF_SIZE - vessel_grid_pos[0])
        local_start_y = max(0, HALF_SIZE - vessel_grid_pos[1])
        
        local_cost[
            int(local_start_x):int(local_start_x + h),
            int(local_start_y):int(local_start_y + w)
        ] = region_cost
        
        # Normalize cost map 
        max_val = local_cost.max()
        if max_val > 0:
            local_cost = local_cost / max_val
        
        return local_cost
    
    def _point_to_line_distance(self, point, line):
        """
        Compute distance from point to line segment
        """
        start, end = line[0], line[1]
        line_vec = end - start
        point_vec = point - start
        
        # Compute projection
        line_length = torch.norm(line_vec)
        if line_length < 1e-6:
            return torch.norm(point - start)
            
        line_unit = line_vec / line_length
        projection = torch.sum(point_vec * line_unit)
        
        # Clamp projection to line segment
        projection = torch.clamp(projection, 0, line_length)
        
        # Compute closest point
        closest = start + line_unit * projection
        
        return torch.norm(point - closest)

    def compute_feature_cost(self, vessel_state):
        """
        Compute cost based on dock features
        """
        cost = torch.zeros_like(self.map_manager.occupancy_grid)
        
        # Distance to dock
        dock_pos = self.state[3:5]
        dist = torch.norm(vessel_state[:2] - dock_pos)
        
        # Orientation alignment
        desired_orientation = self.state[5]
        orientation_error = self._normalize_angle(vessel_state[2] - desired_orientation)
        
        # Safety margin
        safety_cost = self.compute_safety_cost(vessel_state)
        
        # Combine costs
        cost = (self.distance_weight * dist + 
                self.orientation_weight * orientation_error**2 +
                self.safety_weight * safety_cost)
        
        return cost

    def compute_safety_cost(self, vessel_state):
        """
        Compute safety cost based on obstacles
        """
        safety_cost = torch.zeros_like(self.map_manager.occupancy_grid)
        
        # Add exponential cost near obstacles
        occupied = self.map_manager.occupancy_grid > 0.5
        if occupied.any():
            vessel_pos = self.map_manager.world_to_grid(vessel_state[:2].unsqueeze(0))[0]
            for idx in torch.nonzero(occupied):
                dist = torch.norm(vessel_pos.float() - idx.float())
                safety_cost += torch.exp(-dist/2.0)
                
        return safety_cost

    def _normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def update(self, lidar_scan, gps_reading, vessel_pose, control_input, current_time, dt):
        """
        Main update function combining all components
        """
        # Predict step
        self.predict(control_input, dt)
        
        # GPS update
        self.update_gps(gps_reading, current_time)
        
        # Process LiDAR and update map
        features = self.process_lidar(lidar_scan, vessel_pose)
        
        # Update state with features
        self.update_state(features)
        
        # Return cost map for planning
        return self.compute_cost_map(vessel_pose)

    def save_state(self, filename):
        """
        Save estimator state
        """
        state_dict = {
            'state': self.state.cpu(),
            'covariance': self.covariance.cpu(),
            'feature_history': self.feature_history,
            'last_gps_update': self.last_gps_update
        }
        torch.save(state_dict, filename)
        
        # Save map separately
        map_filename = filename.replace('.pt', '_map.pt')
        self.map_manager.save_map(map_filename)

    def load_state(self, filename):
        """
        Load estimator state
        """
        state_dict = torch.load(filename, map_location=self.device)
        self.state = state_dict['state'].to(self.device)
        self.covariance = state_dict['covariance'].to(self.device)
        self.feature_history = state_dict['feature_history']
        self.last_gps_update = state_dict['last_gps_update']
        
        # Load map
        map_filename = filename.replace('.pt', '_map.pt')
        self.map_manager.load_map(map_filename)

    def visualize_state(self):
        """
        Visualize current state estimate
        """
        plt.figure(figsize=(15, 5))
        
        # Plot state estimate
        plt.subplot(131)
        plt.plot(self.state[0].cpu(), self.state[1].cpu(), 'ro', label='Vessel')
        plt.plot(self.state[3].cpu(), self.state[4].cpu(), 'bo', label='Dock')
        plt.arrow(self.state[0].cpu(), self.state[1].cpu(), 
                 np.cos(self.state[2].cpu()), np.sin(self.state[2].cpu()),
                 head_width=0.1, head_length=0.2, fc='r', ec='r')
        plt.arrow(self.state[3].cpu(), self.state[4].cpu(),
                 np.cos(self.state[5].cpu()), np.sin(self.state[5].cpu()),
                 head_width=0.1, head_length=0.2, fc='b', ec='b')
        plt.legend()
        plt.title('State Estimate')
        
        # Plot uncertainty ellipses
        plt.subplot(132)
        self.plot_uncertainty_ellipses()
        plt.title('State Uncertainty')
        
        # Plot cost map
        plt.subplot(133)
        vessel_state = self.state[:3]
        cost_map = self.compute_cost_map(vessel_state)
        plt.imshow(cost_map.cpu(), cmap='hot')
        plt.title('Cost Map')
        
        plt.show()

    def plot_uncertainty_ellipses(self):
        """
        Plot uncertainty ellipses for vessel and dock position
        """
        def plot_ellipse(mean, cov):
            # Ensure consistent dtype
            eigvals, eigvecs = torch.linalg.eigh(cov)
            angle = torch.atan2(eigvecs[1, 0], eigvecs[0, 0])
            
            # Create ellipse points with consistent dtype
            t = np.linspace(0, 2*np.pi, 100)
            x = np.cos(t)
            y = np.sin(t)
            
            # Convert to torch tensors with correct dtype
            xy = torch.tensor([x, y], device=self.device, dtype=torch.float32)
            eigvals = eigvals.to(dtype=torch.float32)
            
            # Scale and rotate ellipse
            R = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                            [torch.sin(angle), torch.cos(angle)]], 
                            device=self.device, 
                            dtype=torch.float32)
            
            # Scale points
            scaled = xy * torch.sqrt(eigvals).unsqueeze(1)
            
            # Rotate points
            rotated = R @ scaled
            
            # Add mean and plot
            mean = mean.to(dtype=torch.float32)
            plt.plot(rotated[0].cpu() + mean[0].cpu(),
                    rotated[1].cpu() + mean[1].cpu())
        
        # Plot vessel uncertainty
        plot_ellipse(self.state[:2], self.covariance[:2, :2])
        
        # Plot dock uncertainty
        plot_ellipse(self.state[3:5], self.covariance[3:5, 3:5])




import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def save_feature_maps(dock_estimator, step, save_dir="feature_maps"):
    """
    Save feature maps extracted during the docking process
    
    Args:
        dock_estimator: HybridDockEstimator instance
        step: Current simulation step
        save_dir: Directory to save feature maps
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Get timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Plot occupancy grid
    plt.subplot(231)
    plt.imshow(dock_estimator.map_manager.occupancy_grid.cpu().numpy(), cmap='binary')
    plt.title('Occupancy Grid')
    plt.colorbar()
    
    # 2. Plot confidence map
    plt.subplot(232)
    plt.imshow(dock_estimator.map_manager.confidence_map.cpu().numpy(), cmap='viridis')
    plt.title('Confidence Map')
    plt.colorbar()
    
    # 3. Plot feature overlay
    plt.subplot(233)
    plt.imshow(dock_estimator.map_manager.occupancy_grid.cpu().numpy(), cmap='binary', alpha=0.5)
    
    # Plot walls
    for wall in dock_estimator.map_manager.dock_features['walls']:
        start, end = wall
        plt.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=2, label='Walls')
        
    # Plot corners
    corners = np.array(dock_estimator.map_manager.dock_features['corners'])
    if len(corners) > 0:
        plt.scatter(corners[:, 0], corners[:, 1], c='g', s=500, label='Corners')
        
    # Plot entrance points
    entrance = np.array(dock_estimator.map_manager.dock_features['entrance_points'])
    if len(entrance) > 0:
        plt.scatter(entrance[:, 0], entrance[:, 1], c='b', s=500, label='Entrance')
    
    plt.title('Feature Overlay')
    plt.legend()
    
    # 4. Plot state estimate
    plt.subplot(234)
    state = dock_estimator.state.cpu().numpy()
    cov = dock_estimator.covariance.cpu().numpy()
    
    # Plot vessel position with uncertainty ellipse
    plot_uncertainty_ellipse(state[:2], cov[:2,:2], color='b', label='Vessel')
    
    # Plot dock position with uncertainty ellipse
    plot_uncertainty_ellipse(state[3:5], cov[3:5,3:5], color='r', label='Dock')
    
    # Plot orientations
    plot_orientation(state[:2], state[2], color='b', length=1.0)
    plot_orientation(state[3:5], state[5], color='r', length=1.0)
    
    plt.title('State Estimate')
    plt.legend()
    
    # 5. Plot cost map
    plt.subplot(235)
    cost_map = dock_estimator.compute_cost_map(torch.tensor(state[:3]))
    plt.imshow(cost_map.cpu().numpy(), cmap='hot')
    plt.title('Cost Map')
    plt.colorbar()
    
    # Save the figure
    filename = f"{save_dir}/feature_map_step_{step}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_uncertainty_ellipse(mean, cov, color='b', label=None):
    """Helper function to plot uncertainty ellipses"""
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.arctan2(eigvecs[1,0], eigvecs[0,0])
    
    # Create ellipse points
    t = np.linspace(0, 2*np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    
    # Stack points into 2xN array
    points = np.vstack([x, y])
    
    # Scale points
    scale_matrix = np.diag(np.sqrt(np.abs(eigvals)))  # Use abs to handle numerical issues
    points = np.dot(scale_matrix, points)
    
    # Create rotation matrix
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    
    # Rotate points
    rotated = np.dot(R, points)
    
    # Translate points
    x_translated = rotated[0, :] + mean[0]
    y_translated = rotated[1, :] + mean[1]
    
    plt.plot(x_translated, y_translated, color=color, label=label)

def plot_orientation(pos, angle, color='b', length=1.0):
    """Helper function to plot orientation arrows"""
    dx = length * np.cos(angle)
    dy = length * np.sin(angle)
    plt.arrow(pos[0], pos[1], dx, dy, head_width=0.1, head_length=0.2, 
             fc=color, ec=color)