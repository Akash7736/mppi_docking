import torch
import os
from bulletsim_aritra_docking import BulletSimulator
# from objective import RoboatObjective, SocialNavigationObjective
from objective import RoboatObjective, SocialNavigationObjective
from dynamics import QuarterRoboatDynamics, WAMVDynamics, WAMVSindyDynamics, QuarterRoboatLSTMDynamics
from ia_mppi.ia_mppi import IAMPPIPlanner
import yaml
from tqdm import tqdm
import copy
import time
from datalogger import DataLogger, generate_random_goals
from docking import * 
import pybullet as p 
from vessel_traj_logger import VesselTrajectoryLogger
from random_weight_gen import WeightRandomizer
from sim_record import SimulationRecorder
from docking_metrics import DockingMetrics

def setup_visualization():
    """Create directories for visualization outputs"""
    os.makedirs('maps', exist_ok=True)



# def run_point_robot_example():
#     data_logger = DataLogger()
#     trajectory_logger = VesselTrajectoryLogger()

#     agent_info = CONFIG['agents']['agent0']  # For agent0, modify as needed
#     # dock_estimator = HybridDockEstimator(device=CONFIG["device"])
#     setup_visualization()
#     dock_estimator = DockEstimator()

# # Initialize the agent goals and device
#     # goals = [
#     #     torch.tensor(agent_info['initial_goal'], device='cuda'),
#     #     torch.tensor(agent_info['goal2'], device='cuda'),
#     #     torch.tensor(agent_info['goal3'], device='cuda'),
#     #     torch.tensor(agent_info['goal4'], device='cuda'),
#     #     torch.tensor(agent_info['goal5'], device='cuda'),
#     # ]
#     dynamics = QuarterRoboatDynamics(
#         cfg=CONFIG
#     )

#     # dynamics = OnrtDynamics(
#     #     cfg=CONFIG
#     # )
#     agent_cost = RoboatObjective(
#         goals=torch.tensor([agent_info['initial_goal'] for agent_info in CONFIG['agents'].values()], device=CONFIG["device"]),
#         device=CONFIG["device"],
#         dock_estimator=dock_estimator 
#     )
#     configuration_cost = SocialNavigationObjective(
#         device=CONFIG["device"]
#     )
        
#     # simulator = Simulator(
#     #     cfg=CONFIG,
#     #     dynamics=dynamics.step
#     # )

#     aritrasim = BulletSimulator(cfg = CONFIG)

#     planner = IAMPPIPlanner(
#         cfg=copy.deepcopy(CONFIG),
#         dynamics=dynamics.step,
#         agent_cost=agent_cost,
#         config_cost=copy.deepcopy(configuration_cost),
#     )

#     initial_action = planner.zero_command()
#     # print(initial_action)
#     # observation = simulator.step(initial_action)
#     observation = aritrasim.step(initial_action)
    

#     current_time = 0.0
#     last_vessel_pose = None

#     for step in tqdm(range(CONFIG['simulator']['steps'])):
#         # print("HELLO")
#         start_time = time.time()

#         # if aritrasim._agents['agent0'].robotId:
#             # Get LiDAR data
#         lidar_distances, lidar_position, lidar_coords = aritrasim._agents['agent0'].get_lidar_data(
#             enable_noise=True,
#             noise_std=0.1
#         )
        
#         lidar_scan = {
#             'ranges': lidar_distances,
#             'angle_increment': np.pi/1800,  # 0.2 degree
#             'angle_min': 0,
#             'angle_max': 2*np.pi,
#             'world_cords': lidar_coords
#         }

#         # Get vessel state
#         pos, orn = p.getBasePositionAndOrientation(aritrasim._agents['agent0'].robotId)
#         euler = p.getEulerFromQuaternion(orn)
#         # In run_point_robot_example()
#         vessel_pose = torch.tensor([pos[0], pos[1], euler[2]], 
#                                 device=CONFIG["device"], 
#                                 dtype=torch.float32)

#         if step % 5 == 0:
#             dock_estimator.lidar_scan =  lidar_scan
#             dock_center, clearances, dock_orientation_angle, world_points, labels, corner_points_world, slopes = dock_estimator.get_dock_center(vessel_pose, step)
#             if (dock_orientation_angle is not None and 
#                 isinstance(dock_center, np.ndarray) and 
#                 dock_center.size > 0 and
#                 isinstance(clearances, (list, np.ndarray)) and 
#                 len(clearances) > 0):
#                 planner._agent_cost.update_dock_state(dock_center, clearances, dock_orientation_angle, world_points, labels,corner_points_world, slopes)
#         # Add GPS noise
#         gps_noise = torch.randn(2, device=CONFIG["device"]) * 2.0  # 2m std dev
#         gps_reading = torch.tensor([pos[0], pos[1]], device=CONFIG["device"]) + gps_noise

#         # Compute control input
#         dt = CONFIG['dt']
#         if last_vessel_pose is not None:
#             dt = CONFIG['dt']
#             control_input = (vessel_pose - last_vessel_pose) / dt
#         else:
#             control_input = torch.zeros(3, device=CONFIG["device"])
#         last_vessel_pose = vessel_pose
#         # local_cost_map = dock_estimator.update(
#         #         lidar_scan=lidar_scan,
#         #         gps_reading=gps_reading,
#         #         vessel_pose=vessel_pose,
#         #         control_input=control_input,
#         #         current_time=current_time,
#         #         dt=dt
#         #     )
            
#         # Update planner cost with new map and estimator state
#         planner._agent_cost.curr_pos = vessel_pose[:2] #aritrasim.agent_position
        
            
#         # planner._agent_cost.local_cost = local_cost_map

#         # Visualize current state periodically
#         # if step % 50 == 0:
#         #     visualize_docking_scenario(
#         #         dock_estimator=dock_estimator,
#         #         vessel_pose=vessel_pose,
#         #         lidar_scan=lidar_scan,
#         #         step=step
#         #     )
#             # save_maps(dock_estimator, step, lidar_scan=lidar_scan)
#             # dock_estimator.visualize_state()




#         planner.make_plan(observation) # Cost calculation and action sequence calculation happens here
#         # end_time = time.time()
#         # print(f"Planning time: {end_time - start_time}")

#         action = planner.get_command()   # action sequence from above is made into dict and return action dict for all agents with first action
#         plans = planner.get_planned_traj() # The action sequence from above is being used to propogate trajectory to horizon
#         # simulator.plot_trajectories(plans)
#         aritrasim.plot_trajectories(plans)

#         # observation = simulator.step(action) # step up all agents with first action sequence and return an observation dict
       
#         observation = aritrasim.step(action) # step up all agents with first action sequence and return an observation dict
#         # print(f"obs {observation} action {action}")
        
#         data_logger.log_data(observation, action)
#         trajectory_logger.log_state(aritrasim._agents['agent0'].robotId, action)
#         # Apply on simulator

#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         sleep_time = CONFIG['dt'] - elapsed_time

#         # if sleep_time > 0:
#         #     time.sleep(sleep_time)

def run_point_robot_example():
    # data_logger = DataLogger()
    trajectory_logger = VesselTrajectoryLogger(config=CONFIG)
    metrics = DockingMetrics(dock_center=(10.0, -5.0), dock_orientation=0.0)

    try:
        agent_info = CONFIG['agents']['agent0']
        setup_visualization()
        dock_estimator = DockEstimator()
        dynamics = QuarterRoboatDynamics(cfg=CONFIG)
        
        agent_cost = RoboatObjective(
            goals=torch.tensor([agent_info['initial_goal'] for agent_info in CONFIG['agents'].values()], device=CONFIG["device"]),
            device=CONFIG["device"],
            cfg=CONFIG, 
            dock_estimator=dock_estimator 
        )
        configuration_cost = SocialNavigationObjective(device=CONFIG["device"])
        
        aritrasim = BulletSimulator(cfg=CONFIG)
        planner = IAMPPIPlanner(
            cfg=copy.deepcopy(CONFIG),
            dynamics=dynamics.step,
            agent_cost=agent_cost,
            config_cost=copy.deepcopy(configuration_cost),
        )

        initial_action = planner.zero_command()
        observation = aritrasim.step(initial_action)
        current_time = 0.0
        last_vessel_pose = None
        recorder = SimulationRecorder(width=1920, height=1080, fps=30)
        recorder.start_recording()
    

        for step in tqdm(range(CONFIG['simulator']['steps'])):
            try:
                start_time = time.time()
                recorder.capture_frame()

                # Get LiDAR data
                lidar_distances, lidar_position, lidar_coords = aritrasim._agents['agent0'].get_lidar_data(
                    enable_noise=True,
                    noise_std=0.1
                )
                
                lidar_scan = {
                    'ranges': lidar_distances,
                    'angle_increment': np.pi/1800,
                    'angle_min': 0,
                    'angle_max': 2*np.pi,
                    'world_cords': lidar_coords
                }

                # Get vessel state
                pos, orn = p.getBasePositionAndOrientation(aritrasim._agents['agent0'].robotId)
                euler = p.getEulerFromQuaternion(orn)
                vessel_pose = torch.tensor([pos[0], pos[1], euler[2]], 
                                        device=CONFIG["device"], 
                                        dtype=torch.float32)

                if step % 5 == 0:
                    dock_estimator.lidar_scan = lidar_scan
                    dock_center, clearances, dock_orientation_angle, world_points, labels, corner_points_world, slopes = dock_estimator.get_dock_center(vessel_pose, step)
                    if (dock_orientation_angle is not None and 
                        isinstance(dock_center, np.ndarray) and 
                        dock_center.size > 0 and
                        isinstance(clearances, (list, np.ndarray)) and 
                        len(clearances) > 0):
                        planner._agent_cost.update_dock_state(dock_center, clearances, dock_orientation_angle, world_points, labels, corner_points_world, slopes)

                # Add GPS noise
                gps_noise = torch.randn(2, device=CONFIG["device"]) * 2.0
                gps_reading = torch.tensor([pos[0], pos[1]], device=CONFIG["device"]) + gps_noise

                # Compute control input
                dt = CONFIG['dt']
                if last_vessel_pose is not None:
                    control_input = (vessel_pose - last_vessel_pose) / dt
                else:
                    control_input = torch.zeros(3, device=CONFIG["device"])
                last_vessel_pose = vessel_pose

                # Update planner
                planner._agent_cost.curr_pos = vessel_pose[:2]
                planner.make_plan(observation)
                action = planner.get_command()
                plans = planner.get_planned_traj()
                aritrasim.plot_trajectories(plans)

                observation = aritrasim.step(action)
                # print(f"OBSSS : {observation['agent0']}")
                col =  aritrasim.check_collision()
                print(f"COLLLLL : {col}")

                if col:
                    print("COLLISION DETECTED !!!")
                    break
                
                # Log trajectory data
                trajectory_logger.log_state(aritrasim._agents['agent0'].robotId, action, observation['agent0'])
                # data_logger.log_data(observation, action)

                end_time = time.time()
                elapsed_time = end_time - start_time
                sleep_time = CONFIG['dt'] - elapsed_time

            except KeyboardInterrupt:
                print("\nUser interrupted the simulation. Saving data and exiting...")
                break

            except Exception as e:
                print(f"\nError in simulation step {step}: {str(e)}")
                print("Attempting to continue simulation...")
                continue

    except Exception as e:
        print(f"\nCritical error in simulation: {str(e)}")
    
    finally:
        # Save all data regardless of how the simulation ended
        try:
            recorder.stop_recording()
            trajectory_logger.save_trajectory_plot(simulator=aritrasim)
            stats = trajectory_logger.get_trajectory_statistics()
            print("\nTrajectory Statistics:")
            for key, value in stats.items():
                print(f"{key}: {value}")
            print("\nTrajectory data has been saved successfully.")
            trajectory_data = trajectory_logger.export_trajectory_data(format='numpy')
            docking_metrics = metrics.calculate_metrics(trajectory_data)
                            # Save performance visualization
            metrics.plot_performance_visualization(trajectory_data)
            plt.savefig(os.path.join(trajectory_logger.log_dir, 
                                    f"docking_performance_{trajectory_logger.timestamp}.png"))
            plt.close()
            # Print docking metrics
            print("\nDocking Performance Metrics:")
            print(f"Success: {docking_metrics['success']}")
            print(f"Final Position Error: {docking_metrics['position_error']:.3f} m")
            print(f"Final Angle Error: {docking_metrics['angle_error_deg']:.2f} degrees")
            print(f"Path Efficiency: {docking_metrics['path_efficiency']:.3f}")
            print(f"Mean Jerk: {docking_metrics['mean_jerk']:.3f}")

            # Save metrics to file
            metrics_file = os.path.join(trajectory_logger.log_dir, 
                                        f"docking_metrics_{trajectory_logger.timestamp}.txt")
            with open(metrics_file, 'w') as f:
                f.write("Docking Performance Metrics:\n")
                for key, value in docking_metrics.items():
                    f.write(f"{key}: {value}\n")
            print(f"\nDocking metrics saved to: {metrics_file}")

        except Exception as e:
            print(f"\nError saving trajectory data: {str(e)}")
        
        try:
            p.disconnect()
            print("PyBullet simulation disconnected.")
        except Exception as e:
            print(f"Error disconnecting PyBullet: {str(e)}")

if __name__ == "__main__":
    # Load the config file
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = f"{abs_path}/cfg_docking.yaml"
    # CONFIG = yaml.safe_load(open(f"{abs_path}/cfg_docking.yaml"))
    # CONFIG = yaml.safe_load(open(f"{abs_path}/cfg_onrt.yaml"))
    dock_center = None 

    # randomizer = WeightRandomizer(f"{abs_path}/cfg_docking.yaml")
    # config_path, weights, initial_pose = randomizer.update_config()
    # print("\nNew configuration:")
    # print(f"Initial pose: x={initial_pose[0]:.2f}, y={initial_pose[1]:.2f}, θ={initial_pose[2]:.2f}")
    # print("\nWeights:")
    # for name, value in weights.items():
    #     print(f"{name}: {value:.4f}")

    # yes = input("if yes: \n")
    # if yes=='y':
    #     pass
    # else:
    #     print("NONONO")


    # Load updated config
    CONFIG = yaml.safe_load(open(config_path))

    run_point_robot_example()


