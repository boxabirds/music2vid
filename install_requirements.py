import platform
import sys
import subprocess
import os
import re

import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description="Process the --dest argument")

# Add the --dest argument with choices
parser.add_argument("--dest", choices=["host", "docker"], default="host", help="Where these dependencies are going: 'host' or 'docker'")

# Parse the arguments
args = parser.parse_args()

# Use the value of the --dest argument
print(f"The destination is: {args.dest}")

install_system_dependencies:bool = False
requirements_file = "non-torch-requirements-host.txt"
#install_scope = ""

if args.dest == "docker":
    requirements_file = "non-torch-requirements-docker.txt"
    #install_scope = "--user"
    install_system_dependencies = True
    print("As we're installing in a docker image, we'll install in --user and also assume elevated privileges and install system dependencies too.")
else:
    print("As we're installing on the host, we'll assume no elevated privileges and not install system dependencies. ")

def is_cuda_12_or_greater(cuda_version):
    major, minor = cuda_version.split('.')
    return int(major) >= 12

def get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        version_line = output.strip().split('\n')[-1]
        version_match = re.search(r"\d+\.\d+(\.\d+)?", version_line)
        if version_match:
            print(f"II CUDA version is: '{version_match.group(0)}'")
            return version_match.group(0)
        else:
            return None
    except FileNotFoundError:
        return None


def do_install_system_dependencies(dependencies):
    try:
        if os.geteuid() != 0:
            print("EE: as this script installs system dependencies it must be run as sudo/root.")
            sys.exit(1)

        subprocess.check_call(["apt-get", "update"])
        install_cmd = ["apt-get", "install", "-y"] + dependencies
        subprocess.check_call(install_cmd)
        print("II System dependencies installed using apt-get.")
    except subprocess.CalledProcessError:
        try:
            subprocess.check_call(["yum", "update", "-y"])
            install_cmd = ["yum", "install", "-y"] + dependencies
            subprocess.check_call(install_cmd)
            print("II System dependencies installed using yum.")
        except subprocess.CalledProcessError as e:
            print(f"EE Error installing system dependencies: {e}")
            sys.exit(1)

os_system = platform.system()
print(f"system detected: {os_system}")

# 13 May 2023: CUDA 12.1 is not supported by pytorch except the nightly build
# cuda / torch dependency management is a mess
def install_torch():
    cuda_version = get_cuda_version()
    if cuda_version:
        print(f"CUDA version is: '{cuda_version}'")
        try:
            if is_cuda_12_or_greater(cuda_version):
                torch_12_install_cmd = [sys.executable, '-m', 'pip', 'install', '--pre', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/nightly/cu121']
                print(f"II torch install: {torch_12_install_cmd}")
                subprocess.check_call(torch_12_install_cmd, shell=False)
        # normal torch install 
            else:
                torch_11_or_earlier_install_cmd = [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio']
                print(f"II torch install for cuda 11 or earlier: {torch_11_or_earlier_install_cmd}")
                subprocess.check_call(torch_11_or_earlier_install_cmd, shell=False)
        except subprocess.CalledProcessError as e: 
            print(f"EE error installing torch: sorry :( {e}")
            sys.exit(1)

    else:
        print("EE CUDA is not installed or nvcc is not in the system path. You're gonna have a bad time. G'bye!")
        sys.exit(1)

def install_remaining_dependencies(os_system):
    print(f"..installing packages from {requirements_file}")
    pip_requirements_txt_cmd = [sys.executable, '-m', 'pip', 'install', '-r', requirements_file]

# For some reason torch needs a specific index url to install https://pytorch.org/get-started/locally/
    torch_extras_if_windows = ['--extra-index-url','https://download.pytorch.org/whl/cu117'] if os_system == 'Windows' else []
    remaining_dependencies_command = pip_requirements_txt_cmd + torch_extras_if_windows
    print(f"II remaining dependencies command: {remaining_dependencies_command}")
    subprocess.call(remaining_dependencies_command, shell=False)


install_torch()
install_remaining_dependencies(os_system)

# Google FILM system dependencies 
if install_system_dependencies:
    system_dependencies = ["libgl1-mesa-glx", "libglib2.0-0"]
    do_install_system_dependencies(system_dependencies)
