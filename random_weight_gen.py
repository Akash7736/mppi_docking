import yaml
import numpy as np
import os
from datetime import datetime

class WeightRandomizer:
    def __init__(self, config_path):
        self.config_path = config_path
        self.base_config = yaml.safe_load(open(config_path))
        self.log_dir = "weight_experiments"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Define reasonable ranges for each weight
        self.weight_ranges = {
            # 'max_speed_weight': (3.0, 7.0),
            # 'back_vel_weight': (0.05, 0.15),
            # 'rot_vel_weight': (0.005, 0.02),
            # 'lat_vel_weight': (0.005, 0.02),
            # 'within_goal_ori_weight': (0.5, 2.0),
            # 'dock_entrance_weight': (2.0, 4.0),
            # 'dock_clearance_weight': (1.5, 3.0),
            # 'dock_goal_weight': (0.5, 2.0),
            # 'dock_heading_to_goal_weight': (2.0, 4.0)
        }
        
        # Dock parameters
        self.dock_position = np.array([10.0, -5.0])  # Dock center position
        self.dock_width = 4.0  # Dock width from config
        self.dock_length = 4.0  # Dock length from config (2m on each side)

        # Define possible spawn regions and their probabilities
        self.spawn_regions = [
            {
                'name': 'front',
                'x_range': (4.0, 8.0),      # Before the dock
                'y_range': (-7.0, -3.0),    # Roughly centered with dock
                'orientation_range': (-np.pi/4, np.pi/4),  # ±45 degrees
                'probability': 0.5
            },
            {
                'name': 'side',
                'x_range': (8.0, 12.0),     # Alongside the dock
                'y_range': (-2.0, 1.0),     # Above the dock
                'orientation_range': (-np.pi/2, 0),  # Facing downward
                'probability': 0.3
            },
            {
                'name': 'far_side',
                'x_range': (8.0, 12.0),     # Alongside the dock
                'y_range': (-8.0, -6.0),    # Below the dock
                'orientation_range': (0, np.pi/2),  # Facing upward
                'probability': 0.2
            }
        ]

    def is_point_in_dock_area(self, x, y):
        """Check if a point is inside or too close to the dock area"""
        # Add safety margin around dock
        safety_margin = 0.5  # 0.5m safety margin
        
        # Dock bounds with safety margin
        dock_x_min = self.dock_position[0] - self.dock_length/2 - safety_margin
        dock_x_max = self.dock_position[0] + self.dock_length/2 + safety_margin
        dock_y_min = self.dock_position[1] - self.dock_width/2 - safety_margin
        dock_y_max = self.dock_position[1] + self.dock_width/2 + safety_margin
        
        # Check if point is in dock area
        return (dock_x_min <= x <= dock_x_max and 
                dock_y_min <= y <= dock_y_max)

    def generate_random_initial_pose(self):
        """Generate random initial pose that is guaranteed to be outside the dock"""
        # Select spawn region based on probabilities
        probabilities = [region['probability'] for region in self.spawn_regions]
        selected_region = np.random.choice(len(self.spawn_regions), p=probabilities)
        region = self.spawn_regions[selected_region]
        
        # Generate position in selected region
        while True:
            x = np.random.uniform(*region['x_range'])
            y = np.random.uniform(*region['y_range'])
            # x = 15
            # y = -5
            x = 10
            y = -9
            
            # Check if position is valid (outside dock area)
            if not self.is_point_in_dock_area(x, y):
                break
        
        # Generate orientation based on region
        orientation = np.random.uniform(*region['orientation_range'])
        
        return [float(x), float(y), float(orientation)]

    def generate_random_weights(self):
        """Generate random weights within defined ranges"""
        new_weights = {}
        for weight_name, (min_val, max_val) in self.weight_ranges.items():
            new_weights[weight_name] = float(np.random.uniform(min_val, max_val))
        return new_weights
    
    def update_config(self):
        """Update config file with new random weights and initial pose"""
        # Generate new weights and pose
        new_weights = self.generate_random_weights()
        new_pose = self.generate_random_initial_pose()
        
        # Update config
        config = self.base_config.copy()
        config['objective'].update(new_weights)
        config['agents']['agent0']['initial_pose'] = new_pose
        
        # Save original config with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_filename = f"config_{timestamp}.yaml"
        config_path = os.path.join(self.log_dir, config_filename)
        
        # Save both copies
        for path in [config_path, self.config_path]:
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        return config_path, new_weights, new_pose
    
    def log_weights(self, weights, initial_pose, performance_metrics=None):
        """Log weights, initial pose, and their performance metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, "weights_log.txt")
        
        with open(log_file, 'a') as f:
            f.write(f"\n\nTimestamp: {timestamp}\n")
            f.write(f"Initial Pose:\n")
            f.write(f"x: {initial_pose[0]:.2f}, y: {initial_pose[1]:.2f}, θ: {initial_pose[2]:.2f}\n")
            f.write("\nWeights:\n")
            for name, value in weights.items():
                f.write(f"{name}: {value:.4f}\n")
            
            if performance_metrics:
                f.write("\nPerformance Metrics:\n")
                for metric, value in performance_metrics.items():
                    f.write(f"{metric}: {value}\n")
            f.write("-" * 50)