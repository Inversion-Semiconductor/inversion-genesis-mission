"""Example usage of the InputParameters class.

This script demonstrates how to use the InputParameters class to manage
and save simulation parameters.
"""

from inversion_fbpic.utils.input_params import InputParameters

# Example 1: Basic usage
def basic_example():
    """Demonstrate basic parameter management."""
    # Clear any existing parameters
    InputParameters.clear()
    
    # Add laser parameters
    InputParameters.add('Laser', {
        'energy_J': 4.0,
        'wavelength_um': 0.800,
        'a0': 2.5,
        'focus_size_m': 25e-6,
    })
    
    # Add gas parameters
    InputParameters.add('Gas', {
        'peak_density_cm-3': 1e18,
        'dopant_level': 0.065,
    })
    
    # Add simulation parameters
    InputParameters.add('Simulation', {
        'nz': 2000,
        'nr': 200,
        'dt_s': 1e-13,
    })
    
    # Print summary
    InputParameters.print_summary()
    
    # Save to INI file
    InputParameters.save_to_ini('output', 'example.ini')


# Example 2: Adding parameters incrementally
def incremental_example():
    """Demonstrate adding parameters incrementally."""
    InputParameters.clear()
    
    # Add some laser parameters
    InputParameters.add('Laser', {
        'energy_J': 4.0,
        'wavelength_um': 0.800,
    })
    
    # Later, add more laser parameters (they will be merged)
    InputParameters.add('Laser', {
        'a0': 2.5,
        'focus_size_m': 25e-6,
    })
    
    # Verify all parameters are stored
    laser_params = InputParameters.get_params('Laser')
    print("Laser parameters:", laser_params)
    
    # Save to file
    InputParameters.save_to_ini('output', 'incremental.ini')


# Example 3: Using in a simulation setup function
def simulation_setup_example():
    """Demonstrate usage within a simulation setup."""
    InputParameters.clear()
    
    # Laser setup
    laser_energy = 4.0
    wavelength = 0.800
    a0 = 2.5
    
    InputParameters.add('Laser', {
        'energy_J': laser_energy,
        'wavelength_um': wavelength,
        'a0': a0,
    })
    
    # Gas setup
    peak_density = 1e18
    dopant_level = 0.065
    
    InputParameters.add('Gas', {
        'peak_density_cm-3': peak_density,
        'dopant_level': dopant_level,
    })
    
    # Save configuration
    save_dir = 'output'
    InputParameters.save_to_ini(save_dir)
    
    print(f"Parameters saved for simulation case")


if __name__ == '__main__':
    print("=== Example 1: Basic Usage ===")
    basic_example()
    
    print("\n=== Example 2: Incremental Addition ===")
    incremental_example()
    
    print("\n=== Example 3: Simulation Setup ===")
    simulation_setup_example()

