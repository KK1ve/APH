import argparse
import os
import warnings
import random
from model import APH
import numpy as np
import torch

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='APH')
parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
parser.add_argument('--optimizer', type=str, default='adam', help="optimizer")
parser.add_argument('--model_dim', type=int, default=4, help="hidden dim should match with x_feature dim")
parser.add_argument('--model', type=str, default='sage', choices=['gcn', 'gat', 'sage', 'cheb'])
parser.add_argument('--seed', type=int, default=42, help="random seed")
parser.add_argument('--task', type=str, default='NPR', choices=['NPR', 'OGR'])
parser.add_argument('--lr', type=float, default=5e-2, help="learning rate")
parser.add_argument('--iteration', type=int, default=8000, help="iteration")
parser.add_argument('--num_heads', type=int, default=4, help="num_heads for GAT")
parser.add_argument('--num_layer', type=int, default=1, help="middle layer")
parser.add_argument('--lam', type=float, default=1, help="lam")
parser.add_argument('--ss', action='store_true', help="Self-Supervise")
parser.add_argument('--ca', action='store_true', help="Cross-Attention")
args = parser.parse_args()

# 固定种子
SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)


# 1, 1/2, 4, 3, 3
# 2, 1, 8, 6, 6
# 1/4, 1/8, 1, 3/4, 3/4
# 1/3, 1/6, 4/3, 1, 1
# 1/3, 1/6, 4/3, 1, 1

def main():
    model = APH(device=device).to(device)
    first = [[1], [torch.tensor(
        [[1, 1 / 3, 1 / 5, 3, 3, 1 / 3], [3, 1, 1 / 2, 5, 6, 1], [5, 2, 1, 6, 7, 3], [1 / 3, 1 / 5, 1 / 6, 1, 1, 1 / 5],
         [1 / 3, 1 / 6, 1 / 7, 1, 1, 1 / 4], [3, 1, 1 / 3, 5, 4, 1]]).to(device)]]
    second = [[6], [torch.tensor([[1, 6, 7, 3, 4, 7, 8], [1 / 6, 1, 1, 1 / 5, 1 / 3, 1, 2], [1 / 7, 1, 1, 1 / 4, 1 / 2, 1, 2],
                            [1 / 3, 5, 4, 1, 2, 4, 7], [1 / 4, 3, 2, 1 / 2, 1, 2, 3], [1 / 7, 1, 1, 1 / 4, 1 / 2, 1, 2],
                            [1 / 8, 1 / 2, 1 / 2, 1 / 7, 1 / 3, 1 / 2, 1]], dtype=torch.float32),

              torch.tensor([[1, 3], [1 / 3, 1]], dtype=torch.float32),
              torch.tensor([[1, 1 / 2, 5, 6, 1, 3, 6], [2, 1, 6, 7, 3, 4, 7], [1 / 5, 1 / 6, 1, 1, 1 / 5, 1 / 3, 1],
                            [1 / 6, 1 / 7, 1, 1, 1 / 4, 1 / 2, 1], [1, 1 / 3, 5, 4, 1, 2, 4],
                            [1 / 3, 1 / 4, 3, 2, 1 / 2, 1, 2],
                            [1 / 6, 1 / 7, 1, 1, 1 / 4, 1 / 2, 1]], dtype=torch.float32),
              torch.tensor([[1, 1 / 7, 1 / 5, 1], [7, 1, 3, 7], [5, 1 / 3, 1, 5], [1, 1 / 7, 1 / 5, 1]],
                           dtype=torch.float32),
              torch.tensor([[1, 3, 5], [1 / 3, 1, 2], [1 / 5, 1 / 2, 1]], dtype=torch.float32),
                             torch.tensor(
                                 [[1, 1 / 2, 3, 6, 4], [2, 1, 3, 7, 4], [1 / 3, 1 / 3, 1, 3, 2],
                                  [1 / 6, 1 / 7, 1 / 3, 1, 1 / 3],
                                  [1 / 4, 1 / 4, 1 / 2, 3, 1]], dtype=torch.float32)
                             ]
              ]
    model(["first", "second"], first=first, second=second)


if __name__ == '__main__':
    main()
