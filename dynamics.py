import torch
import torch.nn as nn

class OmnidirectionalPointRobotDynamics:
    def __init__(self, dt=0.05, device="cuda:0") -> None:
        self._dt = dt
        self._device = device

    def step(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x, y, theta = states[:, :, 0], states[:, :, 1], states[:, :, 2]

        new_x = x + actions[:, :, 0] * self._dt
        new_y = y + actions[:, :, 1] * self._dt
        new_theta = theta + actions[:, :, 2] * self._dt

        new_states = torch.cat([new_x.unsqueeze(2), new_y.unsqueeze(2), new_theta.unsqueeze(2), actions], dim=2)
        return new_states, actions
    
class JackalDynamics:
    def __init__(self, dt=0.05, device="cuda:0") -> None:
        self._dt = dt
        self._device = device
    
    def step(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x, y, theta, vx, vy, omega = states[:, :, 0], states[:, :, 1], states[:, :, 2], states[:, :, 3], states[:, :, 4], states[:, :, 5]

        # Update velocity and position using bicycle model
        new_vx = actions[:, :,0] * torch.cos(theta)
        new_vy = actions[:, :,0] * torch.sin(theta)
        new_omega = actions[:, :,1]

        new_x = x + new_vx * self._dt
        new_y = y + new_vy * self._dt
        new_theta = theta + new_omega * self._dt

        new_states = torch.stack([new_x, new_y, new_theta, new_vx, new_vy, new_omega], dim=2)
        return new_states, actions
    
class QuarterRoboatDynamics:
    def __init__(self, cfg) -> None:
        self.aa = 0.45
        self.bb = 0.90

        self.m11 = 12
        self.m22 = 24
        self.m33 = 1.5
        self.d11 = 6
        self.d22 = 8
        self.d33 = 1.35

        self.cfg = cfg
        self.dt = cfg["dt"]
        self.device = cfg["device"]


        ## Dynamics
        self.D = torch.tensor([		[self.d11	    ,0		,0      ],
                                    [0		,self.d22	    ,0	    ],
                                    [0		,0	    ,self.d33	    ]], device=self.device)

        self.M = torch.tensor([	    [self.m11		,0		,0		],
                                    [0		,self.m22	    ,0		],
                                    [0		,0		,self.m33       ]], device=self.device)
        
        self.B = torch.tensor([	    [1		,1		,0		,0],
                                    [0		,0	    ,1		,1],
                                    [self.aa/2		,-self.aa/2		,self.bb/2    ,-self.bb/2    ]], device=self.device)

        # Inverse of inertia matrix (precalculated for speed)
        self.Minv = torch.inverse(self.M)

    def rot_matrix(self, heading):
        cos = torch.cos(heading).to(self.device)
        sin = torch.sin(heading).to(self.device)
        self.zeros = torch.zeros_like(heading, device=self.device)  # Ensure zeros is on the correct device
        ones = torch.ones_like(heading, device=self.device)

        stacked = torch.stack(
            [cos, -sin, self.zeros, sin, cos, self.zeros, self.zeros, self.zeros, ones],
            dim=1
        ).reshape(heading.size(0), 3, 3, heading.size(1)).to(self.device)

        return stacked.permute(0, 3, 1, 2)

        
    def coriolis(self, vel):
        stacked = torch.stack([self.zeros, self.zeros, -self.m22  * vel[:, :,1], self.zeros, self.zeros, self.m11 * vel[:, :,0], self.m22 * vel[:, :,1], -self.m11 * vel[:, :,0], self.zeros,], dim=1).reshape(vel.size(0), 3, 3, vel.size(1)).to(self.device)

        return stacked.permute(0, 3, 1, 2)
        
    def step(self, states: torch.Tensor, actions: torch.Tensor, t: int = -1) -> torch.Tensor:
        # Ensure all tensors are on the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        
        # Other tensor initializations...
        pose_enu = states[:, :, 0:3]
        pose = torch.zeros_like(pose_enu).to(self.device)
        pose[:, :, 0] = pose_enu[:, :, 1]
        pose[:, :, 1] = pose_enu[:, :, 0]
        pose[:, :, 2] = torch.pi / 2 - pose_enu[:, :, 2]

        vel_enu = states[:, :, 3:6]
        vel = torch.zeros_like(vel_enu).to(self.device)
        vel[:, :, 0] = vel_enu[:, :, 1]
        vel[:, :, 1] = vel_enu[:, :, 0]
        vel[:, :, 2] = -vel_enu[:, :, 2]

        self.zeros = torch.zeros(vel.size(0), vel.size(1), device=self.device)

        # Rotate velocity to the body frame
        vel_body = torch.bmm(
            self.rot_matrix(-pose[:, :, 2]).reshape(-1, 3, 3),
            vel.reshape(-1, 3).unsqueeze(2),
        ).reshape(vel.size(0), vel.size(1), vel.size(2))

        # Ensure other tensors are on the correct device
        Minv_batch = self.Minv.repeat(vel.size(0) * vel.size(1), 1, 1).to(self.device)
        B_batch = self.B.repeat(vel.size(0) * vel.size(1), 1, 1).to(self.device)
        D_batch = self.D.repeat(vel.size(0) * vel.size(1), 1, 1).to(self.device)
        C_batch = self.coriolis(vel_body).reshape(-1, 3, 3).to(self.device)

        new_vel_body = torch.bmm(
            Minv_batch,
            (
                torch.bmm(B_batch, actions.reshape(-1, 4).unsqueeze(2))
                - torch.bmm(C_batch, vel_body.reshape(-1, 3).unsqueeze(2))
                - torch.bmm(D_batch, vel_body.reshape(-1, 3).unsqueeze(2))
            ),
        ).reshape(vel.size(0), vel.size(1), vel.size(2)) * self.dt + vel_body

        # Rotate velocity to the world frame
        vel = torch.bmm(
            self.rot_matrix(pose[:, :, 2]).reshape(-1, 3, 3),
            new_vel_body.reshape(-1, 3).unsqueeze(2),
        ).reshape(vel.size(0), vel.size(1), vel.size(2))

        # Compute new pose
        pose += self.dt * vel

        # Convert from NED to ENU
        new_pose = torch.zeros_like(pose).to(self.device)
        new_pose[:, :, 0] = pose[:, :, 1]
        new_pose[:, :, 1] = pose[:, :, 0]
        new_pose[:, :, 2] = torch.pi / 2 - pose[:, :, 2]
        new_vel = torch.zeros_like(vel).to(self.device)
        new_vel[:, :, 0] = vel[:, :, 1]
        new_vel[:, :, 1] = vel[:, :, 0]
        new_vel[:, :, 2] = -vel[:, :, 2]

        # Set new state
        new_states = torch.concatenate((new_pose, new_vel), 2)

        return new_states, actions


class LSTMDynamicsModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, h=None):
        if h is None:
            h = self.init_hidden(x.size(0))
        out, h = self.lstm(x, h)
        out = self.dropout(out)
        out = self.fc(out)
        return out, h
    
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))

class QuarterRoboatLSTMDynamics:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.dt = cfg["dt"]
        self.device = cfg["device"]
        
        self.model = LSTMDynamicsModel(
            input_size=7,  # [vx, vy, omega, thrust1-4]
            hidden_size=128,
            num_layers=3
        ).to(self.device)
        
        checkpoint = torch.load('models/best_lstm_dynamics.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        scalers = torch.load('models/scalers.pt')
        self.scaler_X = scalers['scaler_X']
        self.scaler_y = scalers['scaler_y']

    def step(self, states: torch.Tensor, actions: torch.Tensor, t: int = -1) -> torch.Tensor:
        batch_size = states.size(0)
        seq_len = states.size(1)
        
        # Extract velocities and scale inputs
        velocities = states[:, :, 3:6]  # [vx, vy, omega]
        x = torch.cat([velocities, actions], dim=2)
        x_reshaped = x.reshape(-1, x.size(-1))
        x_scaled = torch.FloatTensor(self.scaler_X.transform(x_reshaped.cpu())).to(self.device)
        x_scaled = x_scaled.reshape(batch_size, seq_len, -1)
        
        # Get acceleration predictions
        with torch.no_grad():
            derivatives_scaled, _ = self.model(x_scaled)
        
        derivatives = torch.FloatTensor(
            self.scaler_y.inverse_transform(
                derivatives_scaled.cpu().reshape(-1, 3)
            )
        ).to(self.device).reshape(batch_size, seq_len, 3)
        
        # Integrate for new velocities
        new_velocities = velocities + derivatives * self.dt
        
        # Update positions using new velocities
        new_positions = states[:, :, :3] + velocities * self.dt
        
        # Combine positions and velocities
        new_states = torch.cat([new_positions, new_velocities], dim=2)
        
        return new_states, actions

class WAMVDynamics:
    def __init__(self, cfg) -> None:
        # WAM-V physical parameters
        # Distance from COM to thrusters
        self.aa = 1.0  # lateral distance (half-width)
        self.bb = 2.0  # longitudinal distance from COM

        # Inertia matrix parameters (from VRX WAM-V specifications)
        self.m11 = 180.0  # surge mass + added mass
        self.m22 = 220.0  # sway mass + added mass
        self.m33 = 160.0  # yaw moment of inertia + added mass

        # Damping matrix parameters
        self.d11 = 70.0  # surge linear damping
        self.d22 = 100.0  # sway linear damping
        self.d33 = 50.0  # yaw linear damping

        self.cfg = cfg
        self.dt = cfg["dt"]
        self.device = cfg["device"]

        # Dynamics matrices
        # Damping matrix D
        self.D = torch.tensor([
            [self.d11, 0, 0],
            [0, self.d22, 0],
            [0, 0, self.d33]
        ], device=self.device)

        # Inertia matrix M
        self.M = torch.tensor([
            [self.m11, 0, 0],
            [0, self.m22, 0],
            [0, 0, self.m33]
        ], device=self.device)

        # Thruster configuration matrix B
        # For 4 thrusters: [left_front, right_front, left_rear, right_rear]
        self.B = torch.tensor([
            [1, 1, 1, 1],  # X-direction thrust
            [0, 0, 0, 0],  # Y-direction thrust
            [self.aa/2, -self.aa/2, -self.bb/2, self.bb/2]  # Moment around Z
        ], device=self.device)

        # Precalculate inverse of inertia matrix for efficiency
        self.Minv = torch.inverse(self.M)

    def rot_matrix(self, heading):
        """
        Create rotation matrix for given heading angle(s)
        Args:
            heading: tensor of heading angles
        Returns:
            Rotation matrix (batch_size x timesteps x 3 x 3)
        """
        cos = torch.cos(heading)
        sin = torch.sin(heading)
        stacked = torch.stack([
            cos, -sin, self.zeros,
            sin, cos, self.zeros,
            self.zeros, self.zeros, torch.ones_like(heading)
        ], dim=1).reshape(heading.size(0), 3, 3, heading.size(1)).to(self.device)

        return stacked.permute(0, 3, 1, 2)

    def coriolis(self, vel):
        """
        Compute Coriolis matrix C(v)
        Args:
            vel: velocity tensor
        Returns:
            Coriolis matrix (batch_size x timesteps x 3 x 3)
        """
        stacked = torch.stack([
            self.zeros, self.zeros, -self.m22 * vel[:, :, 1],
            self.zeros, self.zeros, self.m11 * vel[:, :, 0],
            self.m22 * vel[:, :, 1], -self.m11 * vel[:, :, 0], self.zeros,
        ], dim=1).reshape(vel.size(0), 3, 3, vel.size(1)).to(self.device)

        return stacked.permute(0, 3, 1, 2)

    def step(self, states: torch.Tensor, actions: torch.Tensor, t: int = -1) -> torch.Tensor:
        """
        Simulate one step of the WAM-V dynamics
        Args:
            states: Current state tensor [x, y, ψ, u, v, r]
            actions: Thruster commands tensor [T1, T2, T3, T4]
            t: Current timestep
        Returns:
            new_states: Updated state tensor
            actions: Applied thruster commands
        """
        # Set current pose and velocity (ENU to NED conversion)
        pose_enu = states[:,:,0:3]
        pose = torch.zeros_like(pose_enu)
        pose[:, :, 0] = pose_enu[:, :, 1]  # x_ned = y_enu
        pose[:, :, 1] = pose_enu[:, :, 0]  # y_ned = x_enu
        pose[:, :, 2] = torch.pi/2 - pose_enu[:, :, 2]  # ψ_ned = π/2 - ψ_enu

        vel_enu = states[:,:,3:6]
        vel = torch.zeros_like(vel_enu)
        vel[:, :, 0] = vel_enu[:, :, 1]  # u_ned = v_enu
        vel[:, :, 1] = vel_enu[:, :, 0]  # v_ned = u_enu
        vel[:, :, 2] = -vel_enu[:, :, 2]  # r_ned = -r_enu

        self.zeros = torch.zeros(vel.size(0), vel.size(1), device=self.device)

        # Transform velocity to body frame
        vel_body = torch.bmm(
            self.rot_matrix(-pose[:,:,2]).reshape(-1,3,3),
            vel.reshape(-1,3).unsqueeze(2)
        ).reshape(vel.size(0), vel.size(1), vel.size(2))

        # Batch matrices for parallel computation
        Minv_batch = self.Minv.repeat(vel.size(0)*vel.size(1), 1, 1)
        B_batch = self.B.repeat(vel.size(0)*vel.size(1), 1, 1)
        D_batch = self.D.repeat(vel.size(0)*vel.size(1), 1, 1)
        C_batch = self.coriolis(vel_body).reshape(-1,3,3)

        # Compute acceleration and integrate for new velocity
        new_vel_body = torch.bmm(
            Minv_batch,
            (torch.bmm(B_batch, actions.reshape(-1,4).unsqueeze(2)) -
             torch.bmm(C_batch, vel_body.reshape(-1,3).unsqueeze(2)) -
             torch.bmm(D_batch, vel_body.reshape(-1,3).unsqueeze(2)))
        ).reshape(vel.size(0), vel.size(1), vel.size(2)) * self.dt + vel_body

        # Transform velocity back to world frame
        vel = torch.bmm(
            self.rot_matrix(pose[:,:,2]).reshape(-1,3,3),
            new_vel_body.reshape(-1,3).unsqueeze(2)
        ).reshape(vel.size(0), vel.size(1), vel.size(2))

        # Integrate velocity for new pose
        pose += self.dt * vel

        # Convert back to ENU
        new_pose = torch.zeros_like(pose)
        new_pose[:, :, 0] = pose[:, :, 1]
        new_pose[:, :, 1] = pose[:, :, 0]
        new_pose[:, :, 2] = torch.pi/2 - pose[:, :, 2]
        
        new_vel = torch.zeros_like(vel)
        new_vel[:, :, 0] = vel[:, :, 1]
        new_vel[:, :, 1] = vel[:, :, 0]
        new_vel[:, :, 2] = -vel[:, :, 2]

        # Combine pose and velocity for complete state
        new_states = torch.concatenate((new_pose, new_vel), 2)

        return new_states, actions
    

import torch

import torch

class WAMVSindyDynamics:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.dt = cfg["dt"]
        self.device = cfg["device"]
        
        # Initialize state constraints with tighter bounds
        self.u_max = 2.0  # Reduced max surge velocity
        self.v_max = 1.0  # Reduced max sway velocity
        self.r_max = 0.5  # Reduced max yaw rate
        
        # Rate constraints for MPC (reduced for stability)
        self.max_accel = 0.5      # Reduced acceleration
        self.max_sway_accel = 0.25 # Reduced sway acceleration
        self.max_yaw_accel = 0.4  # Reduced yaw acceleration

        # Input scaling factor
        self.input_scale = 0.1    # Scale down thruster inputs

        # Additional damping coefficients
        self.surge_damping = 0.8  # Additional surge damping
        self.sway_damping = 0.9   # Additional sway damping
        self.yaw_damping = 0.85   # Additional yaw damping

    def compute_velocity_derivatives(self, velocities, inputs):
        """
        Compute velocity state derivatives using SINDy discovered equations
        with additional stabilization terms
        """
        # Scale inputs
        scaled_inputs = inputs * self.input_scale
        tla, tra, tlf, trf = [scaled_inputs[..., i] for i in range(4)]
        
        # Current velocities
        u, v, r = velocities[..., 0], velocities[..., 1], velocities[..., 2]

        # Add damping terms to velocities
        u_damped = u * self.surge_damping
        v_damped = v * self.sway_damping
        r_damped = r * self.yaw_damping

        # u_dot equation with damping
        u_dot = (-0.112 + 
                 -0.310 * u_damped + -0.063 * v_damped + 0.072 * r_damped + 
                 0.189 * tla + -0.274 * tra + 0.615 * tlf + 0.185 * trf + 
                 -0.048 * u_damped**2 + 0.159 * u_damped * v_damped + -0.093 * u_damped * r_damped +
                 0.423 * u_damped * tla + 0.580 * u_damped * tra + -0.395 * u_damped * tlf + -0.303 * u_damped * trf +
                 0.173 * v_damped**2 + -0.008 * v_damped * r_damped +
                 0.143 * v_damped * tla + 0.037 * v_damped * tra + -0.114 * v_damped * tlf + -0.238 * v_damped * trf +
                 0.026 * r_damped**2 + -0.035 * r_damped * tla + -0.119 * r_damped * tra + 0.126 * r_damped * tlf + 0.055 * r_damped * trf +
                 -0.243 * tla**2 + 0.263 * tla * tra + -0.540 * tla * tlf + -0.058 * tla * trf +
                 0.495 * tra**2 + -0.731 * tra * tlf + -0.575 * tra * trf +
                 0.569 * tlf**2 + 0.465 * tlf * trf + -0.008 * trf**2)

        # v_dot equation with damping
        v_dot = (0.021 + 
                 0.122 * u_damped + 0.106 * v_damped + -0.843 * r_damped +
                 0.213 * tla + 1.203 * tra + -1.333 * tlf + -0.466 * trf +
                 -0.030 * u_damped**2 + -0.468 * u_damped * v_damped + -0.614 * u_damped * r_damped +
                 -0.242 * u_damped * tla + -0.059 * u_damped * tra + -0.052 * u_damped * tlf + 0.145 * u_damped * trf +
                 -0.055 * v_damped**2 + 0.175 * v_damped * r_damped +
                 -0.011 * v_damped * tla + 0.039 * v_damped * tra + -0.147 * v_damped * tlf + -0.108 * v_damped * trf +
                 0.169 * r_damped**2 + 0.709 * r_damped * tla + 0.695 * r_damped * tra + -0.637 * r_damped * tlf + -0.753 * r_damped * trf +
                 -0.748 * tla**2 + 1.861 * tla * tra + -2.963 * tla * tlf + -0.531 * tla * trf +
                 3.280 * tra**2 + -4.484 * tra * tlf + -3.549 * tra * trf +
                 3.960 * tlf**2 + 2.979 * tlf * trf + 0.250 * trf**2)

        # r_dot equation with damping
        r_dot = (0.149 + 
                 0.113 * u_damped + 0.388 * v_damped + -0.493 * r_damped +
                 -0.128 * tla + 2.674 * tra + -2.752 * tlf + -0.165 * trf +
                 -0.001 * u_damped**2 + -0.477 * u_damped * v_damped + -0.260 * u_damped * r_damped +
                 0.408 * u_damped * tla + 0.861 * u_damped * tra + -0.905 * u_damped * tlf + -0.554 * u_damped * trf +
                 0.088 * v_damped**2 + 0.073 * v_damped * r_damped +
                 -0.039 * v_damped * tla + -0.376 * v_damped * tra + 0.249 * v_damped * tlf + -0.063 * v_damped * trf +
                 0.054 * r_damped**2 + 0.385 * r_damped * tla + 0.615 * r_damped * tra + -0.474 * r_damped * tlf + -0.311 * r_damped * trf +
                 -6.947 * tla**2 + -0.646 * tla * tra + -0.494 * tla * tlf + 6.273 * tla * trf +
                 5.878 * tra**2 + -6.805 * tra * tlf + -0.781 * tra * trf +
                 7.387 * tlf**2 + 1.707 * tlf * trf + -5.662 * trf**2)

        return torch.stack([u_dot, v_dot, r_dot], dim=-1)

    def rot_matrix(self, heading):
        """Create rotation matrix for given heading angle(s)"""
        cos = torch.cos(heading)
        sin = torch.sin(heading)
        zeros = torch.zeros_like(heading)
        ones = torch.ones_like(heading)
        
        stacked = torch.stack([
            cos, -sin, zeros,
            sin, cos, zeros,
            zeros, zeros, ones
        ], dim=1).reshape(heading.size(0), 3, 3, heading.size(1)).to(self.device)
        
        return stacked.permute(0, 3, 1, 2)

    def apply_state_constraints(self, states, prev_states=None):
        """Apply state and rate constraints with smoother transitions"""
        # Extract velocities
        velocities = states[..., 3:6]
        
        # Apply velocity constraints with smooth clamping
        velocities_mag = torch.norm(velocities[..., :2], dim=-1, keepdim=True)
        scale = torch.where(
            velocities_mag > self.u_max,
            self.u_max / velocities_mag,
            torch.ones_like(velocities_mag)
        )
        velocities[..., :2] = velocities[..., :2] * scale
        velocities[..., 2] = torch.clamp(velocities[..., 2], -self.r_max, self.r_max)
        
        if prev_states is not None:
            # Rate constraints with smooth transition
            prev_velocities = prev_states[..., 3:6]
            dt = self.dt
            max_du = self.max_accel * dt
            max_dv = self.max_sway_accel * dt
            max_dr = self.max_yaw_accel * dt
            
            delta = velocities - prev_velocities
            delta_mag = torch.norm(delta[..., :2], dim=-1, keepdim=True)
            delta_scale = torch.where(
                delta_mag > max_du,
                max_du / delta_mag,
                torch.ones_like(delta_mag)
            )
            
            delta[..., :2] = delta[..., :2] * delta_scale
            delta[..., 2] = torch.clamp(delta[..., 2], -max_dr, max_dr)
            
            velocities = prev_velocities + delta
        
        # Update velocities in full state
        constrained_states = states.clone()
        constrained_states[..., 3:6] = velocities
        
        return constrained_states

    def step(self, states: torch.Tensor, actions: torch.Tensor, t: int = -1) -> torch.Tensor:
        """
        Simulate one step using RK4 integration with improved stability
        """
        dt = self.dt
        batch_size = states.size(0)
        seq_len = states.size(1)
        
        # Separate pose and velocities
        pose = states[..., :3]
        velocities = states[..., 3:6]
        
        # RK4 integration with smaller intermediate steps
        num_substeps = 4  # Number of substeps for smoother integration
        dt_sub = dt / num_substeps
        
        for _ in range(num_substeps):
            # RK4 integration for current substep
            k1 = self.compute_velocity_derivatives(velocities, actions)
            
            k2_states = self.apply_state_constraints(
                torch.cat([pose, velocities + 0.5 * dt_sub * k1], dim=-1),
                states
            )
            k2 = self.compute_velocity_derivatives(k2_states[..., 3:6], actions)
            
            k3_states = self.apply_state_constraints(
                torch.cat([pose, velocities + 0.5 * dt_sub * k2], dim=-1),
                states
            )
            k3 = self.compute_velocity_derivatives(k3_states[..., 3:6], actions)
            
            k4_states = self.apply_state_constraints(
                torch.cat([pose, velocities + dt_sub * k3], dim=-1),
                states
            )
            k4 = self.compute_velocity_derivatives(k4_states[..., 3:6], actions)
            
            # Update velocities
            new_velocities = velocities + (dt_sub/6) * (k1 + 2*k2 + 2*k3 + k4)
            new_velocities = self.apply_state_constraints(
                torch.cat([pose, new_velocities], dim=-1),
                states
            )[..., 3:6]
            
            # Convert velocities to world frame
            heading = pose[..., 2]
            rot_matrices = self.rot_matrix(heading)
            world_velocities = torch.bmm(
                rot_matrices.reshape(-1, 3, 3),
                new_velocities.reshape(-1, 3, 1)
            ).reshape(batch_size, seq_len, 3)
            
            # Update pose
            pose = pose + dt_sub * world_velocities
            velocities = new_velocities
        
        # Combine final pose and velocities
        new_states = torch.cat([pose, velocities], dim=-1)
        
        # Final constraint check
        new_states = self.apply_state_constraints(new_states, states)
        
        return new_states, actions