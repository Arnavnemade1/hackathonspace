import numpy as np
import unittest
from src.trajectory_optimizer import AdvancedTrajectoryOptimizer

class TestTrajectoryOptimizer(unittest.TestCase):
    def setUp(self):
        """Initialize the trajectory optimizer before each test"""
        self.optimizer = AdvancedTrajectoryOptimizer()
    
    def test_trajectory_calculation(self):
        """Test basic trajectory calculation"""
        mission_params = {
            'initial_position': np.array([0, 0, 0]),
            'initial_velocity': np.array([11.2e3, 0, 0]),
            'mission_duration': 31536000  # 1 year
        }
        
        # Calculate trajectory
        trajectory_data = self.optimizer.calculate_advanced_trajectory(mission_params)
        
        # Assertions
        self.assertIsNotNone(trajectory_data)
        self.assertEqual(len(trajectory_data['trajectory']), 1000)
        self.assertEqual(len(trajectory_data['time_steps']), 1000)
    
    def test_trajectory_shape(self):
        """Verify trajectory data shape"""
        mission_params = {
            'initial_position': np.array([0, 0, 0]),
            'initial_velocity': np.array([11.2e3, 0, 0]),
            'mission_duration': 31536000  # 1 year
        }
        
        trajectory_data = self.optimizer.calculate_advanced_trajectory(mission_params)
        trajectory = trajectory_data['trajectory']
        
        # Check trajectory shape
        self.assertEqual(trajectory.shape[1], 6)  # 6 state variables
    
    def test_gravitational_calculation(self):
        """Test gravitational state derivative calculation"""
        initial_state = np.array([0, 0, 0, 11.2e3, 0, 0])
        derivative = self.optimizer._compute_state_derivative(initial_state)
        
        # Verify derivative calculation
        self.assertEqual(len(derivative), 6)
        self.assertTrue(np.any(derivative != 0))

if __name__ == '__main__':
    unittest.main()
