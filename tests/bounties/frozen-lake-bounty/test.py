import torch
import numpy as np
import gymnasium as gym
import sys

env = gym.make("FrozenLake-v1", is_slippery=False)

np.random.seed(42)

# Load the FrozenLake environment
# Load the TorchScript model
model_path = sys.argv[1]  # Update this path if needed
loaded_model = torch.jit.load(model_path)

# Set the model to evaluation mode
loaded_model.eval()

# Reset the environment
state, _ = env.reset()

steps = 0

# Main loop
for _ in range(100):  # You can adjust the number of steps
    # Convert the state to a torch tensor and add a batch dimension
    state_tensor = torch.tensor([state], dtype=torch.float32).unsqueeze(0)

    # Run the model to get the action
    with torch.no_grad():  # Disable gradient calculation
        outputs = loaded_model(state_tensor)

    action = torch.argmax(outputs[0]).item()  # Get the action with the highest score

    # Take the action in the environment
    next_state, reward, terminated, truncated, _ = env.step(action)
    state = next_state

    if reward == 1:
        break

    steps -= 1

# Close the environment
env.close()

# Print number of steps it took to complete, but negated (fewer steps is better)
print(steps)