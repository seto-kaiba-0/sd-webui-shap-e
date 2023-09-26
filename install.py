import launch
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
req_file = os.path.join(current_dir, "requirements.txt")

with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if not launch.is_installed(lib):
            launch.run_pip(
                f"install {lib}",
                f"sd-webui-shap-e requirement: {lib}")

if not launch.is_installed("diffusers"):
    launch.run_pip(
        f"install diffusers[torch]",
        f"sd-webui-shap-e requirement: huggingface diffusers[torch]")