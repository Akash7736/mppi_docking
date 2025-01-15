import numpy as np
import os
import csv
import time
from datetime import datetime
import torch
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

class VesselTrajectoryLogger:
    def __init__(self, log_dir="trajectory_logs", config=None):
        """Initialize the trajectory logger.
        
        Args:
            log_dir (str): Directory to store trajectory logs
            config: Configuration dictionary containing objective parameters
        """
        self.log_dir = log_dir
        self.config = config
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize data storage
        self.trajectory_data = []
    
    def log_state(self, vessel_id, actions=None, vessel_state=None):
        """Log the current state of the vessel.
        
        Args:
            vessel_id: PyBullet ID of the vessel
            actions: Dictionary containing current control actions
            vessel_state: Full state vector [x, y, theta, vx, vy, omega]
        """
        try:
            if vessel_state is not None:
                if torch.is_tensor(vessel_state):
                    vessel_state = vessel_state.cpu().numpy()
                
                # Extract components from vessel_state
                x, y = vessel_state[0], vessel_state[1]
                theta = vessel_state[2]
                vx, vy = vessel_state[3], vessel_state[4]
                omega = vessel_state[5]
                print(f"OMEGA {omega} theta {theta}")
                
                # Create position and euler arrays
                pos = [x, y, 0.075]  # Fixed z height
                euler = [0.0, 0.0, theta]  # Roll=0, pitch=0, yaw=theta
                
                # Create velocity arrays
                lin_vel = [vx, vy, 0.0]  # z velocity = 0
                ang_vel = [0.0, 0.0, omega]  # Only rotation around z
                
                # Get current timestamp
                current_time = time.time()
                
                # Prepare data row
                data_row = [current_time] + list(pos) + list(euler) + list(lin_vel) + list(ang_vel)

                '''
                    indexing
                    t = 0
                    x = 1, y = 2, z = 3
                    roll = 4, pitch = 5, yaw = 6
                    u = 7, v = 8, w = 9
                    p = 10, q = 11, r = 12
                '''
                
                # Add actions if provided
                if actions is not None and 'agent0' in actions:
                    action_values = actions['agent0'].cpu().numpy().flatten()
                    data_row.extend(action_values)
                else:
                    data_row.extend([0.0] * 4)  # Add zeros if no actions provided
                
                # Store in memory
                self.trajectory_data.append(data_row)
                
        except Exception as e:
            print(f"Error logging state: {str(e)}")

    def plot_trajectory(self, ax, data, simulator=None, bb=True):
        """Plot the vessel trajectory and dock on the given axis.
        
        Args:
            ax: Matplotlib axis to plot on
            data: Trajectory data array
            simulator: BulletSimulator instance containing dock information
        """
        if simulator is not None:
            # Get dock information directly from simulator
            dock_positions = simulator.dock_positions_line1
            dock_width = simulator.dock_width
            
            for i, pos in enumerate(dock_positions):
                # Plot dock center
                # ax.scatter(pos[0], pos[1], color='purple', marker='x', s=200)
                
                # Draw dock walls
                # Back wall
                back_x = pos[0] + 2  # 2m is half the dock length
                ax.plot([back_x, back_x], 
                       [pos[1] - dock_width/2, pos[1] + dock_width/2], 
                       'r-', linewidth=2)
                
                # Side walls
                wall_x = np.array([pos[0] - 2, pos[0] + 2])  # 4m total length
                
                # Left wall
                ax.plot(wall_x, 
                       [pos[1] + dock_width/2, pos[1] + dock_width/2], 
                       'r-', linewidth=2)
                
                # Right wall
                ax.plot(wall_x, 
                       [pos[1] - dock_width/2, pos[1] - dock_width/2], 
                       'r-', linewidth=2)
                
                # Add dock orientation arrow
                orientation = 0 if i < 4 else np.pi
                arrow_length = 1.0
                dx = arrow_length * np.cos(orientation)
                dy = arrow_length * np.sin(orientation)
                ax.arrow(pos[0], pos[1], dx, dy, 
                        head_width=0.2, head_length=0.3, fc='g', ec='g')
        
        # Plot vessel trajectory
                # Plot trajectory with color gradient based on velocity
        velocities = np.linalg.norm(data[:, 7:9], axis=1)
        points = data[:, 1:3]
        sc = ax.scatter(points[:, 0], points[:, 1], c=velocities, 
                   cmap='viridis', s=10)
        
        # ax.plot(data[:, 1], data[:, 2], 'b-', linewidth=2, label='Vessel Path')
        ax.scatter(data[0, 1], data[0, 2], color='g', marker='o', s=100, label='Start')
        ax.scatter(data[-1, 1], data[-1, 2], color='r', marker='o', s=100, label='End')

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax)
        # cbar.set_label('Speed (m/s)')
        cbar.set_label('Speed (m/s)', fontsize=12)  # Change '12' to your preferred size

# Change the font size of the tick labels
        cbar.ax.tick_params(labelsize=10)  
            
        if bb:
            # Plot vessel bounding box at regular intervals
            # Vessel dimensions
            length = 1.0  # Length of the vessel
            width = 0.5   # Width of the vessel
            
            # Calculate corner points in local frame
            corner_points_local = np.array([
                [0.5 * length, 0.5 * width],   # front-right
                [0.5 * length, -0.5 * width],  # back-right
                [-0.5 * length, -0.5 * width], # back-left
                [-0.5 * length, 0.5 * width],  # front-left
                [0.5 * length, 0.5 * width]    # front-right again to close the box
            ])
            
            # Plot vessel outline at regular intervals
            interval = max(len(data) // 10, 1)  # Show about 10 vessel outlines
            for i in range(0, len(data), interval):
                pos = data[i, 1:3]    # x, y position
                theta = data[i,6]    # yaw angle
                
                # Create rotation matrix
                rot_matrix = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])
                
                # Transform corner points to world frame
                corners_world = np.dot(corner_points_local, rot_matrix.T) + pos
                
                # Plot vessel outline
                ax.plot(corners_world[:, 0], corners_world[:, 1], 'k-', alpha=0.3, linewidth=1)
                # Plot forward direction
                ax.arrow(pos[0], pos[1], 
                        0.5 * np.cos(theta), 0.5 * np.sin(theta),
                        head_width=0.1, head_length=0.2, fc='b', ec='b', alpha=0.3)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Vessel Trajectory with Speed Profile (Scenario 3)')
        
        ax.legend()
        ax.grid(True)
        ax.axis('equal')

    def save_trajectory_plot(self, simulator=None):
        """Generate and save plots of the vessel's trajectory and velocities."""
        try:
            # Convert data to numpy array for easier handling
            data = np.array(self.trajectory_data)
            
            if len(data) > 0:
                # Create figure with subplots
                fig = plt.figure(figsize=(20, 12))  # Made taller to accommodate parameters
                
                # Add text for objective parameters
                if self.config is not None and 'objective' in self.config:
                    param_text = (
                        "Objective Parameters:\n"
                        f"max_speed: {self.config['objective']['max_speed']:.2f}  "
                        f"max_speed_weight: {self.config['objective']['max_speed_weight']:.2f}  "
                        f"within_goal_ori_weight: {self.config['objective']['within_goal_ori_weight']:.2f}\n"
                        f"back_vel_weight: {self.config['objective']['back_vel_weight']:.3f}  "
                        f"rot_vel_weight: {self.config['objective']['rot_vel_weight']:.3f}  "
                        f"lat_vel_weight: {self.config['objective']['lat_vel_weight']:.3f}\n"
                        f"dock_entrance_weight: {self.config['objective']['dock_entrance_weight']:.2f}  "
                        f"dock_clearance_weight: {self.config['objective']['dock_clearance_weight']:.2f}  "
                        f"dock_goal_weight: {self.config['objective']['dock_goal_weight']:.2f}  "
                        f"dock_heading_to_goal_weight: {self.config['objective']['dock_heading_to_goal_weight']:.2f}"
                    )
                    # plt.figtext(0.1, 0.95, param_text, fontsize=10, va='top', 
                    #           bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))

                # Create GridSpec with space at top for parameters
                gs = GridSpec(2, 2, figure=fig, top=0.85)
                # gs = GridSpec(1, 2, figure=fig, top=0.85)  # Change to 1 row, 2 columns
                
                # Trajectory plot (top left)
                ax1 = fig.add_subplot(gs[0, 0])
                self.plot_trajectory(ax1, data, simulator)
                
                # Linear velocities plot (top right)
                ax2 = fig.add_subplot(gs[0, 1])
                time_data = data[:, 0] - data[0, 0]  # Time relative to start
                vel = np.sqrt(data[:, 7]**2 + data[:, 8]**2)
                ax2.plot(time_data, data[:, 7], 'b-', label='u (surge)')
                ax2.plot(time_data, data[:, 8], 'r-', label='v (sway)')
                ax2.plot(time_data, data[:, 12], 'g-', label='r (yaw rate)')
                ax2.plot(time_data, vel, 'k-', label='Speed')
                # Add max_speed line if config is available
                if self.config is not None and 'objective' in self.config:
                    max_speed = self.config['objective']['max_speed']
                    ax2.axhline(y=max_speed, color='k', linestyle='--', alpha=0.5, label='Max Speed')
                    ax2.axhline(y=-max_speed, color='k', linestyle='--', alpha=0.5)
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Velocity (m/s)')
                ax2.set_title('Velocities (Scenario 3)')
                ax2.legend()
                ax2.grid(True)
                
                # Angular velocity plot (bottom)
                # ax3 = fig.add_subplot(gs[1, :])
                # ax3.plot(time_data, data[:, 12], 'g-', label='r (yaw rate)')
                # ax3.set_xlabel('Time (s)')
                # ax3.set_ylabel('Angular Velocity (rad/s)')
                # ax3.set_title('Angular Velocity')
                # ax3.legend()
                # ax3.grid(True)
                
                # Adjust layout and save
                plt.tight_layout()
                plot_filename = f"trajectory_plot_{self.timestamp}.png"
                plot_path = os.path.join(self.log_dir, plot_filename)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()
                
                print(f"Saved trajectory plot to: {plot_path}")
                
        except Exception as e:
            print(f"Error saving trajectory plot: {str(e)}")
    
    def get_trajectory_statistics(self):
        """Calculate and return basic statistics about the trajectory."""
        try:
            if not self.trajectory_data:
                return {}
            
            data = np.array(self.trajectory_data)
            
            # Calculate statistics
            stats = {
                'total_distance': np.sum(np.sqrt(np.diff(data[:, 1])**2 + np.diff(data[:, 2])**2)),
                'average_speed': np.mean(np.sqrt(data[:, 7]**2 + data[:, 8]**2)),
                'max_speed': np.max(np.sqrt(data[:, 7]**2 + data[:, 8]**2)),
                'duration': data[-1, 0] - data[0, 0],
                'average_angular_velocity': np.mean(np.abs(data[:, 11])),
                'max_angular_velocity': np.max(np.abs(data[:, 11])),
                'path_start': f"({data[0, 1]:.2f}, {data[0, 2]:.2f})",
                'path_end': f"({data[-1, 1]:.2f}, {data[-1, 2]:.2f})",
                'max_surge_velocity': np.max(np.abs(data[:, 7])),
                'max_sway_velocity': np.max(np.abs(data[:, 8])),
                'rms_yaw_rate': np.sqrt(np.mean(data[:, 12]**2))
            }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating trajectory statistics: {str(e)}")
            return {}
    
    def export_trajectory_data(self, format='numpy'):
        """Export trajectory data in specified format."""
        if not self.trajectory_data:
            return None
            
        if format.lower() == 'numpy':
            return np.array(self.trajectory_data)
        elif format.lower() == 'list':
            return self.trajectory_data
        else:
            raise ValueError("Format must be 'numpy' or 'list'")

    def get_number_of_points(self):
        """Return the number of trajectory points logged."""
        return len(self.trajectory_data)