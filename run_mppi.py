import torch
import os
from bulletsim_aritra_docking import BulletSimulator
# from objective import RoboatObjective, SocialNavigationObjective
from objective import RoboatObjective, SocialNavigationObjective
from dynamics import QuarterRoboatDynamics, WAMVDynamics, WAMVSindyDynamics, QuarterRoboatLSTMDynamics
from ia_mppi import IAMPPIPlanner
import yaml
from tqdm import tqdm
import copy
import time
from datalogger import DataLogger, generate_random_goals
from docking import * 
import pybullet as p 

# Load the config file
abs_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load(open(f"{abs_path}/cfg_docking.yaml"))
# CONFIG = yaml.safe_load(open(f"{abs_path}/cfg_onrt.yaml"))


def run_point_robot_example():
    data_logger = DataLogger()

    agent_info = CONFIG['agents']['agent0']  # For agent0, modify as needed
    dock_estimator = HybridDockEstimator(device=CONFIG["device"])

# Initialize the agent goals and device
    # goals = [
    #     torch.tensor(agent_info['initial_goal'], device='cuda'),
    #     torch.tensor(agent_info['goal2'], device='cuda'),
    #     torch.tensor(agent_info['goal3'], device='cuda'),
    #     torch.tensor(agent_info['goal4'], device='cuda'),
    #     torch.tensor(agent_info['goal5'], device='cuda'),
    # ]
    dynamics = QuarterRoboatDynamics(
        cfg=CONFIG
    )

    # dynamics = OnrtDynamics(
    #     cfg=CONFIG
    # )
    agent_cost = RoboatObjective(
        goals=torch.tensor([agent_info['initial_goal'] for agent_info in CONFIG['agents'].values()], device=CONFIG["device"]),
        device=CONFIG["device"],
        dock_estimator=dock_estimator 
    )
    configuration_cost = SocialNavigationObjective(
        device=CONFIG["device"]
    )
        
    # simulator = Simulator(
    #     cfg=CONFIG,
    #     dynamics=dynamics.step
    # )

    aritrasim = BulletSimulator(cfg = CONFIG)

    planner = IAMPPIPlanner(
        cfg=copy.deepcopy(CONFIG),
        dynamics=dynamics.step,
        agent_cost=agent_cost,
        config_cost=copy.deepcopy(configuration_cost),
    )

    initial_action = planner.zero_command()
    # print(initial_action)
    # observation = simulator.step(initial_action)
    observation = aritrasim.step(initial_action)
    

    current_time = 0.0
    last_vessel_pose = None

    for step in tqdm(range(CONFIG['simulator']['steps'])):
        # print("HELLO")
        start_time = time.time()

        if aritrasim._agents['agent0'].robotId:
            # Get LiDAR data
            lidar_distances, lidar_position, lidar_coords = aritrasim._agents['agent0'].get_lidar_data(
                enable_noise=True,
                noise_std=0.1
            )
            
            lidar_scan = {
                'ranges': lidar_distances,
                'angle_increment': np.pi/1800,  # 0.2 degree
                'angle_min': 0,
                'angle_max': 2*np.pi
            }

            # Get vessel state
            pos, orn = p.getBasePositionAndOrientation(aritrasim._agents['agent0'].robotId)
            euler = p.getEulerFromQuaternion(orn)
           # In run_point_robot_example()
            vessel_pose = torch.tensor([pos[0], pos[1], euler[2]], 
                                    device=CONFIG["device"], 
                                    dtype=torch.float32)

            # Add GPS noise
            gps_noise = torch.randn(2, device=CONFIG["device"]) * 2.0  # 2m std dev
            gps_reading = torch.tensor([pos[0], pos[1]], device=CONFIG["device"]) + gps_noise

            # Compute control input
            if last_vessel_pose is not None:
                dt = CONFIG['dt']
                control_input = (vessel_pose - last_vessel_pose) / dt
            else:
                control_input = torch.zeros(3, device=CONFIG["device"])
            last_vessel_pose = vessel_pose

            # Update dock estimator
            cost_map = dock_estimator.update(
                lidar_scan=lidar_scan,
                gps_reading=gps_reading,
                vessel_pose=vessel_pose,
                control_input=control_input,
                current_time = current_time,
                dt=CONFIG['dt']
            )

            # Update planner cost with new map and estimator state
            planner._agent_cost.curr_pos = aritrasim.agent_position
            planner._agent_cost.update_dock_state(dock_estimator.state, dock_estimator.covariance)
            planner._agent_cost.local_cost = cost_map

            # Visualize current state periodically
            if step % 50 == 0:
                save_feature_maps(dock_estimator, step)
                # dock_estimator.visualize_state()




        planner.make_plan(observation) # Cost calculation and action sequence calculation happens here
        # end_time = time.time()
        # print(f"Planning time: {end_time - start_time}")

        action = planner.get_command()   # action sequence from above is made into dict and return action dict for all agents with first action
        plans = planner.get_planned_traj() # The action sequence from above is being used to propogate trajectory to horizon
        # simulator.plot_trajectories(plans)
        aritrasim.plot_trajectories(plans)

        # observation = simulator.step(action) # step up all agents with first action sequence and return an observation dict
       
        observation = aritrasim.step(action) # step up all agents with first action sequence and return an observation dict
        print(f"obs {observation} action {action}")
        
        data_logger.log_data(observation, action)
        # Apply on simulator

        end_time = time.time()
        elapsed_time = end_time - start_time
        sleep_time = CONFIG['dt'] - elapsed_time

        # if sleep_time > 0:
        #     time.sleep(sleep_time)


if __name__ == "__main__":
    run_point_robot_example()
