# run_all.py
import os
import sys
def main():
    print("Starting descriptive audio generation (combine_script.py)...")
    ret1 = os.system('python3 combine_script.py')
    if ret1 != 0:
        print("Error running combine_script.py. Exiting.")
        sys.exit(1)
    print("\nRunning video modification with audio insertion (Logic_script.py)...")
    ret2 = os.system('python3 Logic_script.py')
    if ret2 != 0:
        print("Error running Logic_script.py. Exiting.")
        sys.exit(1)
    print("\nProcess completed successfully.")
if __name__ == "__main__":
    main()