# Telecommunication system

## Overview
This repository provides a Python implementation of an `Antenna` class for simulating signal transmission, reception, and power feasibility in a telecom environment. The class extends `DynamicCylinder` from the `omni.isaac.core.objects` module and includes raycasting functionality to determine signal coherence between antennas.

## Features
The `Antenna` class includes the following functionalities:

### 1. **Antenna Properties & Initialization**
- `waveLength`: Wavelength of the emitted/received signal.
- `G`: Antenna gain.
- `e`: Antenna efficiency.
- `directivity`: Calculated as `G/e`.
- `frequency`: Computed as `C / waveLength`, where `C` is the speed of light.
- `mode`: `1` for emitter, `0` for receiver.
- `power`: Minimum detectable power for the antenna.

### 2. **Antenna Configuration**
- `setMode(mode)`: Sets the antenna as either an emitter (`1`) or receiver (`0`).
- `setDirection(theta, phi)`: Defines the emission/reception direction.
- `setPower(powerValue)`: Sets the minimum receivable power.

### 3. **Raycasting for Signal Transmission**
- `performRaycast(target_antenna, stage, distance_increment)`:
  - Casts a ray from the emitter antenna to the receiver to determine if an object blocks the signal.
  - Returns a tuple `(hit_object_path, distance)`.

- `moveOriginRaycast(target_antenna, increment)`:
  - Adjusts the raycast origin slightly forward for better accuracy.

### 4. **Signal Coherence Check**
- `checkRaycastCoherence(output_ray, target_path)`:
  - Checks if the raycast hit the expected target antenna.
  - Returns `True` if successful, otherwise `False`.

- `displayRaycastCoherence(output_ray, target_path)`:
  - Prints whether the signal reached the target or not.

### 5. **Signal Power Computation**
- `power_losses(distance)`:
  - Computes signal loss over a given distance based on the inverse-square law.

- `checkPowerFeasibility(distance, minimum_power)`:
  - Determines if the received power is above the threshold.

- `displayPowerFeasibility(output_ray, target_antenna)`:
  - Prints whether the signal is strong enough for detection.

## Usage Example
```python
# Create an emitter and receiver antenna
emitter = Antenna("/World/emitter", "Emitter", waveLength=0.1, G=10, e=0.8, mode=1, power=0.01)
receiver = Antenna("/World/receiver", "Receiver", waveLength=0.1, G=8, e=0.75, mode=0)

# Set receiver position
receiver.set_world_pose(position=np.array([10, 10, 23]))

# Perform raycast
stage = get_current_stage()
output_ray = emitter.performRaycast(receiver, stage, 0.3)

# Check if signal reached the receiver
receiver.displayRaycastCoherence(output_ray, receiver.prim_path)
receiver.displayPowerFeasibility(output_ray, receiver)
```

## Dependencies
- `numpy`
- `scipy`
- `omni.isaac.core.objects`
- `omni.physx`
- `pxr`

## License
This project is licensed under the MIT License. Feel free to use and modify it as needed.

---

For more details, feel free to open an issue or contribute to the repository!

