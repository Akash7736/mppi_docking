import torch 
import numpy as np 
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt 
from sklearn.mixture import GaussianMixture



class DockEstimator:
    def __init__(self):
        self.lidar_scan = None
        self.lidar_points = None
        self.dock_center = None
        

    def get_lidar_points(self):
        angles = np.arange(self.lidar_scan['angle_min'], self.lidar_scan['angle_max'], self.lidar_scan['angle_increment'])
        # Convert ranges to a NumPy array for easier comparison
        ranges = np.array(self.lidar_scan['ranges'])

        # Filter out distances greater than 40
        valid_ranges = ranges <= 40
        valid_ranges_indices = np.where(valid_ranges)[0]

        # Apply filter to ranges, angles, and compute x, y coordinates
        filtered_ranges = ranges[valid_ranges_indices]
        filtered_angles = angles[valid_ranges_indices]

        # Calculate the x and y positions based on the filtered ranges and angles
        x = filtered_ranges * np.cos(filtered_angles)
        y = filtered_ranges * np.sin(filtered_angles)


        # Create a point cloud from lidar scan
        self.lidar_points = np.vstack((x, y)).T  # Shape: (N, 2), where N is the number of lidar measurements
        # plt.scatter(self.lidar_points[:, 0], self.lidar_points[:, 1], s=10, label='Lidar Points')
        # plt.show()
        # plt.close()
        return self.lidar_points

    def get_dock_center(self, vessel_pose, step=None):
        vessel_pose = vessel_pose.cpu().numpy()
        vessel_position = vessel_pose[:2]

        vessel_position = vessel_pose[:2]  # Vessel position (center)
        vessel_orientation = vessel_pose[2]  # Vessel orientation (yaw)

        # Vessel dimensions
        length = 1.5  # Length of the vessel (1 meter)
        width = 1.0  # Width of the vessel (0.5 meter)
        
        # Calculate 4 corner points of the vessel in the local frame (no orientation yet)
        corner_points_local = np.array([
            [0.5 * length, 0.5 * width],  # front-right
            [0.5 * length, -0.5 * width],  # back-right
            [-0.5 * length, 0.5 * width],  # front-left
            [-0.5 * length, -0.5 * width]   # back-left
        ])

        # Create the rotation matrix for the vessel's orientation
        rotation_matrix = np.array([
            [np.cos(vessel_orientation), -np.sin(vessel_orientation)],
            [np.sin(vessel_orientation), np.cos(vessel_orientation)]
        ])
        
        # Rotate the corner points from local frame to world frame
        corner_points_rotated = np.dot(corner_points_local, rotation_matrix.T)
        
        # Translate the rotated corner points by the vessel's position
        corner_points_world = corner_points_rotated + vessel_position


        self.get_lidar_points()
        self.clustering = DBSCAN(eps=0.5, min_samples=10).fit(self.lidar_points)

        labels = self.clustering.labels_
        dock_points = self.lidar_points[labels != -1]  # Extract points belonging to the docking station
        # Calculate the centroid (middle point) of the docking station
        self.dock_center = np.mean(dock_points, axis=0)
        world_points = self.transform_to_world(self.lidar_points, vessel_pose)
        n_components = 3  # Number of walls
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(world_points)
        labels = gmm.predict(world_points)

                # Store line parameters and wall centers
        line_parameters = []  # [(slope, intercept), ...]
        wall_centers = []  # [(x, y), ...]
        clearances = []  # Clearance distances from the vessel to each wall
        slopes = []
        plt.close()

        # Step 3: Fit RANSAC lines for each cluster
        for i in range(n_components):
            # Get cluster points
            cluster_points = world_points[labels == i]
            
            # Extract x and y coordinates
            X = cluster_points[:, 0].reshape(-1, 1)  # Reshape for RANSAC
            y = cluster_points[:, 1]
            
            # Fit line using RANSAC
            ransac = RANSACRegressor(random_state=42)
            ransac.fit(X, y)
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_
            line_parameters.append((slope, intercept))
            
            # Predict line points for visualization
            x_line = np.linspace(X.min(), X.max(), 100)
            y_line = slope * x_line + intercept
            
            # Plot cluster points and fitted line
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Wall {i+1}')
            plt.plot(x_line, y_line, color='red', linestyle='--', label=f'Line {i+1}')
            
            # Compute wall center (mean of points)
            wall_center = cluster_points.mean(axis=0)
            wall_centers.append(wall_center)

            # Plot wall center
            plt.scatter(*wall_center, color='orange', marker='o', s=100, label=f'Wall Center {i+1}')

            # Step 4: Compute clearance
            x0, y0 = vessel_position
            m, c = slope, intercept
            clearance = abs(m * x0 - y0 + c) / np.sqrt(m**2 + 1)
            slopes.append((m,c))
            # clearances.append(clearance)
            # distances = np.linalg.norm(cluster_points - vessel_position, axis=1)
            distances = np.linalg.norm(cluster_points[:, np.newaxis] - corner_points_world, axis=2)

        # Get the minimum distance for each corner point to any point in the cluster
            min_distances_for_corners = distances.min(axis=0)
        
        # Get the minimum distance for the cluster
            min_distance = distances.min()
            # clearances.append(min_distance)
            clearances.append(min_distances_for_corners.min())

        # Step 5: Compute dock center
        dock_center = np.mean(wall_centers, axis=0)

        # Step 6: Identify parallel walls and compute dock orientation
        tolerance = 0.1  # Adjust based on your data
        parallel_wall_indices = []
        for i, (slope1, _) in enumerate(line_parameters):
            for j, (slope2, _) in enumerate(line_parameters):
                if i < j and abs(slope1 - slope2) < tolerance:
                    parallel_wall_indices = [i, j]
                    break
            if parallel_wall_indices:
                break

        try:
            if parallel_wall_indices:
                # Compute the angles of the parallel walls
                angle1 = np.arctan(line_parameters[parallel_wall_indices[0]][0])
                angle2 = np.arctan(line_parameters[parallel_wall_indices[1]][0])
                
                # Average the angles for dock orientation
                dock_orientation_angle = (angle1 + angle2) / 2
                entrance_point = self.compute_entrance_point(dock_center, dock_orientation_angle)
                plt.scatter(*entrance_point, color='magenta', marker='*', s=200, label='Entrance Point')
        
            else:
                raise ValueError("Could not find parallel walls!")
        except (IndexError, ValueError) as e:
            # Handle errors if wall indices are missing or invalid
            print(f"Error computing dock orientation: {e}")
            dock_orientation_angle = None  # Default value if computation fails

            # import os
            # os.makedirs("lidar", exist_ok=True)
            # np.save(f'lidar/points_{step}.npy',self.lidar_points)

        # Step 7: Visualize dock center, orientation, and vessel clearance
        plt.scatter(*dock_center, color='purple', marker='x', s=150, label='Dock Center')
        
        
        plt.quiver(
            dock_center[0], dock_center[1], 
            np.cos(dock_orientation_angle), np.sin(dock_orientation_angle),
            color='green', scale=5, width=0.005, label='Dock Orientation'
        )
        plt.scatter(*vessel_position, color='blue', marker='s', s=100, label='Vessel Position')
        for point in corner_points_world:
            plt.scatter(point[0], point[1])

        # Step 8: Final plot settings
        plt.title('Dock Segmentation, Orientation, and Vessel Clearance')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')  # This ensures proper aspect ratio
        

        # Output results
        print("Line Parameters (slope, intercept):", line_parameters)
        print("Wall Centers:", wall_centers)
        print("Dock Center:", dock_center)
        print("Dock Orientation Angle (degrees):", np.degrees(dock_orientation_angle))
        print(f"corner points : {corner_points_world}")
        print("Clearances from Vessel to Walls:", clearances)
        print(f"vessel pose {vessel_position} ")
        print(f"entrance point {entrance_point}")
        # plt.show()

        return dock_center, clearances, dock_orientation_angle, world_points, labels, entrance_point, slopes


    def transform_to_world(self, points: np.ndarray, vessel_pose: np.ndarray) -> np.ndarray:
        """Transform points from vessel to world frame"""
        # Ensure vessel_pose and points are of float32 type
        vessel_pose = vessel_pose.astype(np.float32)
        points = points.astype(np.float32)
        
        cos_theta = np.cos(vessel_pose[2])
        sin_theta = np.sin(vessel_pose[2])
        
        # Rotation matrix
        R = np.array([[cos_theta, -sin_theta],
                    [sin_theta, cos_theta]], dtype=np.float32)
        
        # Apply transformation
        transformed_points = (R @ points.T).T + vessel_pose[:2]
        
        return transformed_points 
    
    def compute_entrance_point(self, dock_center, dock_orientation_angle):
        """
        Compute entrance point 2m away from dock center in the opposite direction of dock orientation.
        
        Args:
            dock_center (np.ndarray): The center point of the dock [x, y]
            dock_orientation_angle (float): The orientation angle of the dock in radians
        
        Returns:
            np.ndarray: The entrance point coordinates [x, y]
        """
        # Distance from dock center to entrance point
        entrance_distance = 2.0  # meters
        
        # Add π to get the opposite direction of dock orientation
        entrance_angle = dock_orientation_angle + np.pi
        
        # Calculate entrance point coordinates
        entrance_x = dock_center[0] + entrance_distance * np.cos(entrance_angle)
        entrance_y = dock_center[1] + entrance_distance * np.sin(entrance_angle)
        
        entrance_point = np.array([entrance_x, entrance_y])
        
        return entrance_point