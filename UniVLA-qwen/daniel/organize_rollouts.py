#%%
import os

folder = "/fs/scratch/PAS2099/danielf/geometry_grounded_latents/UniVLA/rollouts/2025_10_03"
file_ls = sorted(os.listdir(folder))

save_dir = "/fs/scratch/PAS2099/danielf/geometry_grounded_latents/UniVLA/rollouts/organized"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for file in file_ls:
    task_name = (file.split("--task=")[1]).split(".")[0]
    task_dir = os.path.join(save_dir, task_name)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    success = "success=True" in file

    if success:
        final_dir = os.path.join(task_dir, "success")
    else:
        final_dir = os.path.join(task_dir, "failure")

    if not os.path.exists(final_dir):
        os.makedirs(final_dir)

    os.system(f"mv {os.path.join(folder, file)} {final_dir}")

