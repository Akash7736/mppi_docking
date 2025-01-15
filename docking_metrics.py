import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

class DockingMetrics:
    def __init__(self, dock_center=(10.0, -5.0), dock_orientation=0.0):
        """Initialize docking metrics calculator.
        
        Args:
            dock_center (tuple): (x, y) coordinates of dock center
            dock_orientation (float): Orientation of dock in radians
        """
        self.dock_center = np.array(dock_center)
        self.dock_orientation = dock_orientation
        
        # Define success criteria thresholds
        self.position_threshold = 0.3    # meters from dock center
        self.angle_threshold = np.pi/12  # 15 degrees from dock orientation
        self.velocity_threshold = 0.1    # m/s for final approach
        self.angular_velocity_threshold = 0.1  # rad/s for final approach
        
    def calculate_metrics(self, trajectory_data):
        """Calculate comprehensive docking metrics.
        
        Args:
            trajectory_data: numpy array containing trajectory data
                [timestamp, x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, ...]
        
        Returns:
            dict: Dictionary containing all docking metrics
        """
        # Extract final state
        final_pos = trajectory_data[-1, 1:3]
        final_orientation = trajectory_data[-1, 6]  # yaw
        final_vel = trajectory_data[-1, 7:9]       # vx, vy
        final_angular_vel = trajectory_data[-1, 12] # wz

        # Calculate basic metrics
        position_error = np.linalg.norm(final_pos - self.dock_center)
        angle_error = abs(((final_orientation - self.dock_orientation + np.pi) % (2*np.pi)) - np.pi)
        final_speed = np.linalg.norm(final_vel)
        
        # Calculate success criteria
        position_success = position_error <= self.position_threshold
        angle_success = angle_error <= self.angle_threshold
        velocity_success = final_speed <= self.velocity_threshold
        angular_success = abs(final_angular_vel) <= self.angular_velocity_threshold
        
        overall_success = (position_success and angle_success and 
                         velocity_success and angular_success)

        # Calculate trajectory smoothness
        velocity_data = trajectory_data[:, 7:9]  # vx, vy
        angular_velocity = trajectory_data[:, 12]  # wz
        
        # Jerk metrics (rate of change of acceleration)
        dt = np.diff(trajectory_data[:, 0])  # Time steps
        acceleration = np.diff(velocity_data, axis=0) / dt[:, np.newaxis]
        jerk = np.diff(acceleration, axis=0) / dt[:-1, np.newaxis]
        mean_jerk = np.mean(np.linalg.norm(jerk, axis=1))
        
        # Time and efficiency metrics
        total_time = trajectory_data[-1, 0] - trajectory_data[0, 0]
        path_length = np.sum(np.linalg.norm(np.diff(trajectory_data[:, 1:3], axis=0), axis=1))
        direct_distance = np.linalg.norm(trajectory_data[-1, 1:3] - trajectory_data[0, 1:3])
        path_efficiency = direct_distance / path_length if path_length > 0 else 0
        
        # Stability metrics
        lateral_deviation = self.calculate_lateral_deviation(trajectory_data)
        orientation_stability = np.std(trajectory_data[:, 5])  # std of yaw
        velocity_stability = np.std(np.linalg.norm(velocity_data, axis=1))

        return {
            # Success metrics
            'success': overall_success,
            'position_error': float(position_error),
            'angle_error_deg': float(np.degrees(angle_error)),
            'final_speed': float(final_speed),
            'final_angular_velocity': float(final_angular_vel),
            
            # Time and efficiency metrics
            'total_time': float(total_time),
            'path_length': float(path_length),
            'path_efficiency': float(path_efficiency),
            
            # Smoothness metrics
            'mean_jerk': float(mean_jerk),
            'lateral_deviation': float(lateral_deviation),
            'orientation_stability': float(orientation_stability),
            'velocity_stability': float(velocity_stability),
            
            # Individual success criteria
            'position_success': bool(position_success),
            'angle_success': bool(angle_success),
            'velocity_success': bool(velocity_success),
            'angular_success': bool(angular_success)
        }
    
    def calculate_lateral_deviation(self, trajectory_data):
        """Calculate RMS lateral deviation from ideal approach path."""
        # Define ideal approach path (straight line to dock)
        start_pos = trajectory_data[0, 1:3]
        approach_vector = self.dock_center - start_pos
        approach_direction = approach_vector / np.linalg.norm(approach_vector)
        
        # Calculate lateral deviations
        positions = trajectory_data[:, 1:3]
        deviations = []
        for pos in positions:
            # Vector from start to current position
            current_vector = pos - start_pos
            # Project onto approach vector
            projection = np.dot(current_vector, approach_direction) * approach_direction
            # Calculate lateral deviation
            deviation = np.linalg.norm(current_vector - projection)
            deviations.append(deviation)
        
        return np.sqrt(np.mean(np.array(deviations)**2))

    def plot_performance_visualization(self, trajectory_data):
        """Create comprehensive visualization of docking performance."""
        plt.figure(figsize=(15, 10))
        
        # Plot trajectory with color gradient based on velocity
        velocities = np.linalg.norm(trajectory_data[:, 7:9], axis=1)
        points = trajectory_data[:, 1:3]
        
        plt.subplot(221)
        plt.scatter(points[:, 0], points[:, 1], c=velocities, 
                   cmap='viridis', s=10)
        plt.colorbar(label='Velocity (m/s)')
        plt.plot([self.dock_center[0]], [self.dock_center[1]], 'r*', 
                 markersize=15, label='Dock')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Trajectory with Velocity Profile')
        plt.axis('equal')
        plt.grid(True)
        
        # Plot velocity profile
        plt.subplot(222)
        time = trajectory_data[:, 0] - trajectory_data[0, 0]
        plt.plot(time, velocities, label='Speed')
        plt.plot(time, trajectory_data[:, 11], label='Angular Velocity')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity')
        plt.title('Velocity Profiles')
        plt.grid(True)
        plt.legend()
        
        # Plot heading error
        plt.subplot(223)
        heading_error = np.degrees(((trajectory_data[:, 5] - self.dock_orientation + np.pi) 
                                  % (2*np.pi)) - np.pi)
        plt.plot(time, heading_error)
        plt.xlabel('Time (s)')
        plt.ylabel('Heading Error (degrees)')
        plt.title('Heading Error vs Time')
        plt.grid(True)
        
        # Plot distance to dock
        plt.subplot(224)
        distances = np.linalg.norm(trajectory_data[:, 1:3] - self.dock_center, axis=1)
        plt.plot(time, distances)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance to Dock (m)')
        plt.title('Distance to Dock vs Time')
        plt.grid(True)
        
        plt.tight_layout()
        return plt.gcf()

    def compile_experiment_results(self, all_metrics):
        """Compile statistics across multiple experimental runs.
        
        Args:
            all_metrics: List of metric dictionaries from different runs
            
        Returns:
            dict: Aggregate statistics
        """
        metrics_array = {key: [] for key in all_metrics[0].keys()}
        for metrics in all_metrics:
            for key, value in metrics.items():
                metrics_array[key].append(value)
        
        # Calculate aggregate statistics
        aggregate_stats = {}
        
        # Success rate
        aggregate_stats['success_rate'] = np.mean([m['success'] for m in all_metrics])
        
        # Numerical metrics
        numerical_metrics = ['position_error', 'angle_error_deg', 'final_speed', 
                           'total_time', 'path_efficiency', 'mean_jerk',
                           'lateral_deviation', 'orientation_stability']
        
        for metric in numerical_metrics:
            values = metrics_array[metric]
            aggregate_stats[f'{metric}_mean'] = np.mean(values)
            aggregate_stats[f'{metric}_std'] = np.std(values)
            aggregate_stats[f'{metric}_min'] = np.min(values)
            aggregate_stats[f'{metric}_max'] = np.max(values)
        
        return aggregate_stats
        
    def format_results_table(self, aggregate_stats):
        """Format results for publication-ready table."""
        table_rows = [
            "Metric & Mean ± Std & Min & Max \\\\",
            "\\hline",
            f"Success Rate & {aggregate_stats['success_rate']*100:.1f}\\% & - & - \\\\",
            f"Position Error (m) & {aggregate_stats['position_error_mean']:.3f} ± {aggregate_stats['position_error_std']:.3f} & {aggregate_stats['position_error_min']:.3f} & {aggregate_stats['position_error_max']:.3f} \\\\",
            f"Angle Error (deg) & {aggregate_stats['angle_error_deg_mean']:.2f} ± {aggregate_stats['angle_error_deg_std']:.2f} & {aggregate_stats['angle_error_deg_min']:.2f} & {aggregate_stats['angle_error_deg_max']:.2f} \\\\",
            f"Completion Time (s) & {aggregate_stats['total_time_mean']:.2f} ± {aggregate_stats['total_time_std']:.2f} & {aggregate_stats['total_time_min']:.2f} & {aggregate_stats['total_time_max']:.2f} \\\\",
            f"Path Efficiency & {aggregate_stats['path_efficiency_mean']:.3f} ± {aggregate_stats['path_efficiency_std']:.3f} & {aggregate_stats['path_efficiency_min']:.3f} & {aggregate_stats['path_efficiency_max']:.3f} \\\\",
            f"Mean Jerk & {aggregate_stats['mean_jerk_mean']:.3f} ± {aggregate_stats['mean_jerk_std']:.3f} & {aggregate_stats['mean_jerk_min']:.3f} & {aggregate_stats['mean_jerk_max']:.3f} \\\\"
        ]
        
        return "\n".join(table_rows)