import csv
import time
from datetime import datetime
import os
import torch
import random
import csv
import time

class DataLogger:
    def __init__(self, output_dir="logs"):
        os.makedirs(output_dir, exist_ok=True)
        self.filename = f"{output_dir}/sim_data_{int(time.time())}.csv"
        self.start_time = time.time()
        self.initialize_csv()
        
    def initialize_csv(self):
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['time_s', 'x', 'y', 'theta', 'vx', 'vy', 'omega', 
                      'thruster1', 'thruster2', 'thruster3', 'thruster4']
            writer.writerow(headers)
            
    def log_data(self, observation, action):
        elapsed_time = time.time() - self.start_time
        state = observation['agent0']
        actions = action['agent0']
        
        row_data = [f"{elapsed_time:.3f}"] + state.tolist() + actions.tolist()
        
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
            
def generate_random_goals(device, n_goals=10):
    # Initial predefined goals
 
    goals = []
    # Add random goals to make total of 10
    while len(goals) < n_goals:
        # Generate random x,y coordinates within safe bounds (-8 to 8)
        # Avoid areas too close to docks (around x=±10)
        x = random.uniform(-20, 20)
        y = random.uniform(-20, 20)
        theta = random.uniform(-3.14, 3.14)  # Random orientation
        
        # Create goal tensor
        random_goal = torch.tensor([x, y, theta], device=device)
        goals.append(random_goal)
    
    return goals