import os
import sys
import subprocess
import shutil

def build_exe():
    if not os.path.exists("logo.ico"):
        print("错误：未找到logo.ico文件")
        return
    
    try:
        import PyInstaller
    except ImportError:
        print("正在安装PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--icon=logo.ico",
        "--name=ExVR-xingfu",
        "--add-data=models;models",
        "--add-data=tracker;tracker",
        "--add-data=utils;utils",
        "main.py"
    ]
    
    print("开始编译...")
    print(f"命令：{' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        print("编译完成！")
        print("可执行文件位于dist/ExVR.exe")
    except subprocess.CalledProcessError as e:
        print(f"编译失败：{e}")

if __name__ == "__main__":
    build_exe()
