"""Official inference entry, deterministic initial RNG and loopback-only socket."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch,random,numpy as np
# Explicit initial RNG only; official model/config/inference remain unchanged.
torch.manual_seed(0);torch.cuda.manual_seed_all(0);random.seed(0);np.random.seed(0)
from wan_va.configs import VA_CONFIGS
VA_CONFIGS['robotwin'].host='127.0.0.1'
from wan_va.wan_va_server import main,init_logger
if __name__=='__main__':
 init_logger()
 main()
