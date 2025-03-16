import subprocess
import sys

def install_package(package):
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        with open("failed_packages.txt", "a") as f:
            f.write(f"{package}\n")

def main(requirements_file):
    with open(requirements_file, "r") as file:
        packages = file.readlines()

    for package in packages:
        package = package.strip()
        if package and not package.startswith("#"):
            install_package(package)

if __name__ == "__main__":
    requirements_file = "D:\\requirements.txt"
    main(requirements_file)