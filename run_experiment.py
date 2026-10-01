"""Entry point kept small so spawned GPU workers can safely import modules."""
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "2")
os.environ.setdefault("XFORMERS_DISABLED", "1")

from postcode_ml.experiment import main

if __name__ == "__main__":
    main()
