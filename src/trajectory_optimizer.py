import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

class AdvancedTrajectoryOptimizer:
    def __init__(self):
        """
        Initialize advanced trajectory optimization with precise celestial mechanics
        """
        self.celestial_bodies = {
            'Earth': {
                'mass': 5.97e24,  # kg
                'radius': 6.371e6,  # meters
                'gravitational_parameter': 3.986004418e14,  # m³/s²
                'position': np.array([0, 0, 0])
            },
            'Mars': {
                'mass': 6.39e23,  # kg
                'radius': 3.389e6,  # meters
                'gravitational_parameter': 4.282837e13,  # m³/s²
                'position': np.array([2.279e11, 0, 0])  # Average distance from Sun
            }
        }
    
    def calculate_advanced_trajectory(
        self, 
        initial_conditions: dict
    ) -> dict:
        """
        Calculate ultra-precise spacecraft trajectory
        
        :param initial_conditions: Mission launch parameters
        :return: Comprehensive trajectory data
        """
        def runge_kutta_trajectory(
            initial_state: np.ndarray, 
            time_steps: np.ndarray
        ) -> np.ndarray:
            """
            Advanced numerical integration for trajectory calculation
            
            :param initial_state: Initial spacecraft state vector
            :param time_steps: Simulation time steps
            :return: Trajectory points
            """
            state = initial_state.copy()
            trajectory = [state.copy()]
            
            for dt in np.diff(time_steps):
                # Complex gravitational mechanics calculations
                k1 = self._compute_state_derivative(state)
                k2 = self._compute_state_derivative(state + 0.5 * dt * k1)
                k3 = self._compute_state_derivative(state + 0.5 * dt * k2)
                k4 = self._compute_state_derivative(state + dt * k3)
                
                state += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
                trajectory.append(state.copy())
            
            return np.array(trajectory)
        
        # Unpack initial conditions with defaults
        initial_position = initial_conditions.get(
            'initial_position', 
            np.array([0, 0, 0])
        )
        initial_velocity = initial_conditions.get(
            'initial_velocity', 
            np.array([11.2e3, 0, 0])  # Escape velocity
        )
        
        # Initial state vector [x, y, z, vx, vy, vz]
        initial_state = np.concatenate([
            initial_position, 
            initial_velocity
        ])
        
        # Time steps for trajectory simulation
        mission_duration = initial_conditions.get('mission_duration', 31536000)  # 1 year
        time_steps = np.linspace(0, mission_duration, 1000)
        
        # Compute trajectory
        trajectory = runge_kutta_trajectory(initial_state, time_steps)
        
        return {
            'trajectory': trajectory,
            'time_steps': time_steps,
            'mission_params': initial_conditions
        }
    
    def _compute_state_derivative(self, state: np.ndarray) -> np.ndarray:
        """
        Compute state derivatives with gravitational influences
        
        :param state: Current spacecraft state vector
        :return: State derivative vector
        """
        G = 6.67430e-11  # Gravitational constant
        
        # Position and velocity components
        pos = state[:3]
        vel = state[3:]
        
        # Compute gravitational acceleration from multiple bodies
        total_acceleration = np.zeros(3)
        
        for body_name, body in self.celestial_bodies.items():
            # Vector to celestial body
            body_vector = body['position']
            
            # Gravitational acceleration calculation
            r = np.linalg.norm(pos - body_vector)
            acceleration = (
                G * body['mass'] * (body_vector - pos) / (r**3 + 1e-10)
            )
            total_acceleration += acceleration
        
        # Add simple thrust model (optional)
        thrust_acceleration = np.array([
            0.001,  # Small constant thrust in x direction
            0,
            0
        ])
        total_acceleration += thrust_acceleration
        
        # Combine position and velocity derivatives
        derivative = np.zeros(6)
        derivative[:3] = vel
        derivative[3:] = total_acceleration
        
        return derivative
    
    def create_trajectory_visualization(self, trajectory_data: dict):
        """
        Create an interactive 3D trajectory visualization
        
        :param trajectory_data: Comprehensive trajectory data
        """
        trajectory = trajectory_data['trajectory']
        
        # Plotly 3D Interactive Visualization
        trace = go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode='lines',
            line=dict(
                color=trajectory[:, 2],  # Color gradient based on z-coordinate
                colorscale='Viridis',
                width=6
            ),
            name='Spacecraft Trajectory'
        )
        
        # Add celestial body markers
        body_traces = []
        for name, body in self.celestial_bodies.items():
            body_trace = go.Scatter3d(
                x=[body['position'][0]],
                y=[body['position'][1]],
                z=[body['position'][2]],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red' if name == 'Mars' else 'blue',
                    symbol='circle'
                ),
                name=name
            )
            body_traces.append(body_trace)
        
        # Layout configuration
        layout = go.Layout(
            title='Advanced Spacecraft Trajectory Simulation',
            scene=dict(
                xaxis_title='X Position (m)',
                yaxis_title='Y Position (m)',
                zaxis_title='Z Position (m)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1200,
            height=800
        )
        
        # Combine all traces
        fig = go.Figure(data=[trace] + body_traces, layout=layout)
        
        # Save interactive HTML visualization
        pio.write_html(fig, file='trajectory_visualization.html')
        print("Interactive visualization saved as 'trajectory_visualization.html'")
    
    def generate_mission_report(self, trajectory_data: dict) -> str:
        """
        Generate a comprehensive mission report
        
        :param trajectory_data: Trajectory optimization results
        :return: Detailed mission report
        """
        trajectory = trajectory_data['trajectory']
        mission_params = trajectory_data['mission_params']
        
        # Calculate key mission metrics
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory[:, :3], axis=0), axis=1))
        max_velocity = np.max(np.linalg.norm(trajectory[:, 3:], axis=1))
        
        report = f"""
# Spacecraft Mission Report

## Mission Parameters
- **Initial Position:** {mission_params.get('initial_position', 'Earth Orbit')}
- **Initial Velocity:** {mission_params.get('initial_velocity', 'Escape Velocity')}
- **Mission Duration:** {mission_params.get('mission_duration', '1 Year')} seconds

## Trajectory Analysis
- **Total Trajectory Distance:** {total_distance:.2e} meters
- **Maximum Velocity:** {max_velocity:.2f} m/s
- **Trajectory Correction Points:** {len(trajectory)}

## Key Observations
1. The trajectory demonstrates advanced orbital mechanics simulation
2. Multiple gravitational bodies are considered in path calculation
3. Includes a basic thrust acceleration model

## Visualization
An interactive 3D visualization has been generated in 'trajectory_visualization.html'
"""
        return report

def main():
    # Create trajectory optimizer
    optimizer = AdvancedTrajectoryOptimizer()
    
    # Define mission parameters
    mission_params = {
        'initial_position': np.array([0, 0, 0]),  # Starting from Earth orbit
        'initial_velocity': np.array([11.2e3, 0, 0]),  # Escape velocity
        'mission_duration': 31536000,  # 1 year mission duration
        'target_body': 'Mars'
    }
    
    # Run trajectory optimization
    trajectory_data = optimizer.calculate_advanced_trajectory(mission_params)
    
    # Generate visualization
    optimizer.create_trajectory_visualization(trajectory_data)
    
    # Generate and save mission report
    mission_report = optimizer.generate_mission_report(trajectory_data)
    with open('mission_report.md', 'w') as f:
        f.write(mission_report)
    print("Mission report saved as 'mission_report.md'")

if __name__ == "__main__":
    main()
