import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import os

class EnhancedTrajectoryVisualizer:
    def __init__(self, optimizer):
        """
        Initialize enhanced trajectory visualization
        
        :param optimizer: The trajectory optimizer instance
        """
        self.optimizer = optimizer
        
    def generate_animation_frames(self, trajectory_data, num_frames=100):
        """
        Generate animation frames for the trajectory
        
        :param trajectory_data: Comprehensive trajectory data
        :param num_frames: Number of animation frames to generate
        :return: List of animation frames
        """
        trajectory = trajectory_data['trajectory']
        time_steps = trajectory_data['time_steps']
        
        # Calculate indices for frames
        indices = np.linspace(0, len(trajectory) - 1, num_frames, dtype=int)
        
        frames = []
        for i, idx in enumerate(indices):
            # Calculate completion percentage for color gradient
            completion = i / (num_frames - 1)
            
            # Only show trajectory up to current point
            current_traj = trajectory[:idx+1]
            
            # Create frame for spacecraft position
            spacecraft_trace = go.Scatter3d(
                x=[trajectory[idx, 0]],
                y=[trajectory[idx, 1]],
                z=[trajectory[idx, 2]],
                mode='markers',
                marker=dict(
                    size=8,
                    color='gold',
                    symbol='diamond',
                    line=dict(color='black', width=1)
                ),
                name='Spacecraft'
            )
            
            # Create frame for trajectory path
            trajectory_trace = go.Scatter3d(
                x=current_traj[:, 0],
                y=current_traj[:, 1],
                z=current_traj[:, 2],
                mode='lines',
                line=dict(
                    color=np.arange(len(current_traj)),
                    colorscale='Viridis',
                    width=5
                ),
                name='Trajectory Path'
            )
            
            # Create frame data
            frame_data = [spacecraft_trace, trajectory_trace]
            
            # Add celestial bodies to each frame
            for name, body in self.optimizer.celestial_bodies.items():
                # Calculate body size for visualization (not to scale but proportional)
                relative_size = 15 if name == 'Earth' else 10
                
                body_trace = go.Scatter3d(
                    x=[body['position'][0]],
                    y=[body['position'][1]],
                    z=[body['position'][2]],
                    mode='markers',
                    marker=dict(
                        size=relative_size,
                        color='blue' if name == 'Earth' else 'red',
                        opacity=0.8,
                        symbol='sphere'
                    ),
                    name=name
                )
                frame_data.append(body_trace)
            
            # Add time information
            current_time = time_steps[idx]
            days = current_time / (24 * 3600)
            
            # Add metrics for current frame
            current_velocity = np.linalg.norm(trajectory[idx, 3:])
            distance_from_earth = np.linalg.norm(trajectory[idx, :3])
            
            # Add annotation for time
            time_annotation = go.Scatter3d(
                x=[trajectory[idx, 0]],
                y=[trajectory[idx, 1]],
                z=[trajectory[idx, 2] + 1e7],  # Position above spacecraft
                mode='text',
                text=[f"T+{days:.1f} days"],
                textposition="top center",
                name='Mission Time'
            )
            frame_data.append(time_annotation)
            
            frames.append(dict(
                name=str(i),
                data=frame_data,
                traces=[0, 1, 2, 3, 4]  # Match indices with the data list
            ))
        
        return frames
    
    def create_enhanced_visualization(self, trajectory_data):
        """
        Create an enhanced interactive 3D trajectory visualization with animation
        
        :param trajectory_data: Comprehensive trajectory data
        :return: HTML string with the visualization
        """
        trajectory = trajectory_data['trajectory']
        
        # Initial empty figure for animation
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'scatter3d'}]],
            subplot_titles=["Spacecraft Trajectory: Earth to Mars Mission"]
        )
        
        # Initial trajectory trace (empty)
        trajectory_trace = go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode='lines',
            line=dict(
                colorscale='Viridis',
                width=5
            ),
            name='Trajectory Path'
        )
        
        # Initial spacecraft position (empty)
        spacecraft_trace = go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode='markers',
            marker=dict(
                size=8,
                color='gold',
                symbol='diamond',
                line=dict(color='black', width=1)
            ),
            name='Spacecraft'
        )
        
        # Add celestial bodies
        for name, body in self.optimizer.celestial_bodies.items():
            # Calculate body size for visualization (not to scale but proportional)
            relative_size = 15 if name == 'Earth' else 10
            color = 'royalblue' if name == 'Earth' else 'firebrick' 
            
            body_trace = go.Scatter3d(
                x=[body['position'][0]],
                y=[body['position'][1]],
                z=[body['position'][2]],
                mode='markers',
                marker=dict(
                    size=relative_size,
                    color=color,
                    opacity=0.9,
                    symbol='sphere'
                ),
                name=name
            )
            fig.add_trace(body_trace)
        
        # Add initial time annotation
        time_annotation = go.Scatter3d(
            x=[trajectory[0, 0]],
            y=[trajectory[0, 1]],
            z=[trajectory[0, 2] + 1e7],
            mode='text',
            text=["T+0.0 days"],
            textposition="top center",
            name='Mission Time'
        )
        
        # Add traces to figure
        fig.add_trace(spacecraft_trace)
        fig.add_trace(trajectory_trace)
        fig.add_trace(time_annotation)
        
        # Generate animation frames
        frames = self.generate_animation_frames(trajectory_data, num_frames=100)
        fig.frames = frames
        
        # Set up animation settings
        animation_settings = dict(
            frame=dict(duration=100, redraw=True),
            fromcurrent=True,
            mode="immediate",
            transition=dict(duration=50, easing="cubic-in-out"),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0.1,
                    xanchor="right",
                    yanchor="top",
                    pad=dict(t=0, r=10),
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, animation_settings]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=0)
                                )
                            ]
                        )
                    ]
                )
            ]
        )
        
        # Update figure layout with animation settings
        fig.update_layout(
            title="Earth to Mars Spacecraft Trajectory Simulation",
            scene=dict(
                xaxis_title="X Position (m)",
                yaxis_title="Y Position (m)", 
                zaxis_title="Z Position (m)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='data'
            ),
            updatemenus=[animation_settings["updatemenus"][0]],
            sliders=[
                dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue=dict(
                        font=dict(size=16),
                        prefix="Mission Day: ",
                        visible=True,
                        xanchor="right"
                    ),
                    transition=dict(duration=300, easing="cubic-in-out"),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [str(i)],
                                dict(
                                    frame=dict(duration=100, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=300)
                                )
                            ],
                            label=str(int(i * trajectory_data['time_steps'][-1] / (24 * 3600 * 100)))
                        )
                        for i in range(100)
                    ]
                )
            ],
            width=1000,
            height=800,
            margin=dict(l=0, r=0, b=0, t=30),
            template="plotly_dark"
        )
        
        # Save as standalone HTML file
        html_file = 'enhanced_trajectory_visualization.html'
        pio.write_html(fig, file=html_file, auto_open=False, include_plotlyjs='cdn')
        
        return html_file
    
    def create_dashboard_app(self, trajectory_data):
        """
        Create a Dash application for trajectory visualization and data display
        
        :param trajectory_data: Comprehensive trajectory data
        :return: Dash application instance
        """
        # Extract data
        trajectory = trajectory_data['trajectory']
        time_steps = trajectory_data['time_steps']
        
        # Create Dash app
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        
        # Generate the enhanced visualization HTML
        vis_html_file = self.create_enhanced_visualization(trajectory_data)
        with open(vis_html_file, 'r') as f:
            vis_html = f.read()
        
        # Calculate mission metrics
        distances = np.linalg.norm(trajectory[:, :3], axis=1)
        velocities = np.linalg.norm(trajectory[:, 3:], axis=1)
        time_days = time_steps / (24 * 3600)
        
        # Create app layout
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Space Trajectory Optimization Dashboard", 
                           className="text-center mt-4 mb-4")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("3D Trajectory Visualization"),
                        dbc.CardBody([
                            html.Iframe(
                                srcDoc=vis_html,
                                style={"width": "100%", "height": "600px", "border": "none"}
                            )
                        ])
                    ])
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Mission Parameters"),
                        dbc.CardBody([
                            html.Div([
                                html.H5("Initial Conditions"),
                                html.P(f"Position: {trajectory_data['mission_params'].get('initial_position', 'Earth Orbit')}"),
                                html.P(f"Velocity: {trajectory_data['mission_params'].get('initial_velocity', 'Escape Velocity')} m/s"),
                                html.P(f"Duration: {trajectory_data['mission_params'].get('mission_duration', '1 Year')/86400:.1f} days"),
                                
                                html.H5("Current Metrics", className="mt-4"),
                                html.Div(id="current-metrics"),
                                
                                html.H5("Mission Controls", className="mt-4"),
                                dbc.Input(
                                    id="time-slider", 
                                    type="range", 
                                    min=0, 
                                    max=len(time_days)-1, 
                                    value=0,
                                    className="mt-2"
                                ),
                                html.Div(id="time-display", className="text-center mt-2")
                            ])
                        ])
                    ], className="mb-4"),
                    
                    dbc.Card([
                        dbc.CardHeader("Telemetry Data"),
                        dbc.CardBody([
                            dcc.Graph(
                                id="telemetry-chart",
                                figure={
                                    "data": [
                                        {
                                            "x": time_days,
                                            "y": distances,
                                            "type": "line",
                                            "name": "Distance from Origin (m)"
                                        },
                                        {
                                            "x": time_days,
                                            "y": velocities,
                                            "type": "line",
                                            "name": "Velocity (m/s)"
                                        }
                                    ],
                                    "layout": {
                                        "title": "Mission Telemetry",
                                        "xaxis": {"title": "Time (days)"},
                                        "yaxis": {"title": "Value"},
                                        "legend": {"x": 0, "y": 1},
                                        "template": "plotly_dark"
                                    }
                                }
                            )
                        ])
                    ])
                ], width=4)
            ])
        ], fluid=True)
        
        # Callback to update metrics based on slider
        @app.callback(
            [Output("current-metrics", "children"),
             Output("time-display", "children")],
            [Input("time-slider", "value")]
        )
        def update_metrics(time_idx):
            if time_idx is None:
                time_idx = 0
                
            current_position = trajectory[time_idx, :3]
            current_velocity = trajectory[time_idx, 3:]
            current_time = time_steps[time_idx]
            
            distance_from_earth = np.linalg.norm(current_position)
            speed = np.linalg.norm(current_velocity)
            days = current_time / (24 * 3600)
            
            # Calculate distance to Mars
            mars_pos = self.optimizer.celestial_bodies['Mars']['position']
            distance_to_mars = np.linalg.norm(current_position - mars_pos)
            
            metrics = [
                html.P(f"Distance from Earth: {distance_from_earth/1000:.2f} km"),
                html.P(f"Distance to Mars: {distance_to_mars/1000:.2f} km"),
                html.P(f"Current Speed: {speed:.2f} m/s"),
                html.P(f"Days Elapsed: {days:.2f}")
            ]
            
            time_display = f"Mission Time: T+{days:.2f} days"
            
            return metrics, time_display
        
        return app
        
    def run_dashboard(self, trajectory_data, port=8050):
        """
        Run the trajectory visualization dashboard
        
        :param trajectory_data: Trajectory data dictionary
        :param port: Port to run the server on
        """
        app = self.create_dashboard_app(trajectory_data)
        print(f"Starting dashboard server on port {port}...")
        print(f"Open your browser to http://localhost:{port} to view the dashboard")
        app.run_server(debug=False, port=port, host='0.0.0.0')


def main():
    # Import the optimizer here to avoid circular imports
    from src.trajectory_optimizer import AdvancedTrajectoryOptimizer
    
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
    
    # Create enhanced visualizer and run dashboard
    visualizer = EnhancedTrajectoryVisualizer(optimizer)
    visualizer.run_dashboard(trajectory_data)

if __name__ == "__main__":
    main()
