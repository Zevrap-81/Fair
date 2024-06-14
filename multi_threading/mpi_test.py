from mpi4py import MPI
from tqdm import tqdm
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Function for subprocess to update progress
def subprocess_task(sub_rank):
    progress = 0
    # Some task that increments progress
    for i in tqdm(range(100), desc=f"Subprocess {sub_rank}"):
        time.sleep(0.1)
        progress += 1
    return progress

# Main process
if rank == 0:
    sub_progresses = []

    # Launch subprocesses
    for sub_rank in range(1, size):
        comm.send(True, dest=sub_rank)  # Signal to start subprocess
        sub_progresses.append(comm.recv(source=sub_rank))  # Receive progress from subprocess

    # Gather progress from subprocesses
    total_progress = sum(sub_progresses)
    print(f"Total progress across subprocesses: {total_progress}")

# Subprocesses
else:
    while True:
        start_task = comm.recv(source=0)  # Receive signal to start task from main process
        if start_task:
            progress = subprocess_task(rank)  # Execute task in subprocess
            comm.send(progress, dest=0)  # Send progress back to main process
            break  # Break subprocess loop after completing task
