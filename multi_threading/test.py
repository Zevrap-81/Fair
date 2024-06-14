import time

from computing.compare_gpu import compare_skill_arr_method_2
from computing.model import load_base_model

# Generate some test data
# Replace this with your actual data
skill_eval_arr = [{'skill': 'Python', 'score': 0.8}, {'skill': 'TensorFlow', 'score': 0.9}]
base_arr = [{'skill': 'Python', 'score': 0.7}, {'skill': 'PyTorch', 'score': 0.85}]

# Record the start time
start_time = time.time()

# Call your function
model = load_base_model()
result = compare_skill_arr_method_2(model, skill_eval_arr, base_arr)  # Replace with your function call

# Calculate the execution time
execution_time = time.time() - start_time

# Print the result and execution time
print(f"Result: {result}")
print(f"Execution time: {execution_time} seconds")