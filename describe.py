import git
import sys
import os
import subprocess
print("name of run")
run_name = input()
run_dir = "results/" + run_name + "/"
os.mkdir(run_dir)

with open(run_dir + "info.txt", "w") as f:
    f.write("Command:\n")
    f.write(" ".join(sys.argv) + "\n")
    f.write("System:\n")
    f.write(subprocess.check_output(["hostname"]).decode())
    f.write("Git Hash:\n")
    f.write(subprocess.check_output(["git", "describe", "--always"]).strip().decode() + "\n")
    f.write("Uncommitted changes:\n")
    f.write(subprocess.check_output(["git", "diff", "HEAD"]).decode())
    
