from typing import List, Dict

import torch


class APH(torch.nn.Module):
    def __init__(self, **kwargs):
        super(APH, self).__init__()
        assert "device" in kwargs
        self.device = kwargs["device"]

    def forward(self, *args, **kwargs) -> Dict:
        """
        :param args:  层级列表 从最大层级开始  e.g. [metrics_one, metrics_two]
        :param kwargs:  模糊综合判断矩阵 e.g. {metrics_one: List[torch.Tensor], metrics_two: List[torch.Tensor]}

        :return:

        """
        args = args[0]

        result_dict = {args[i]: [self.consistency_checks(z) for z in kwargs[args[i]]] for i in range(len(args))}

        print(result_dict)
        return result_dict


    def consistency_checks(self, matrix: torch.Tensor) -> Dict:
        n, m = matrix.shape

        # 因素指标权重计算
        w_i: torch.Tensor = torch.zeros(n, 1)
        for i in range(n):
            w_i[i] = torch.pow(torch.prod(matrix[i]), 1 / n)
        W = w_i / torch.sum(w_i, dim=0)     # shape = n,1

        # 求判断矩阵最大特征值
        _ = torch.mm(matrix, W)
        eigen_max = torch.max(torch.sum(_ / W, dim=0) / n)

        # 一致性指标 CI
        CI = (eigen_max - n) / (n - 1)

        # RI指标
        RI_list = [0., 0., 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
        if RI_list[n-1] != 0.:
            RI = torch.tensor(RI_list[n-1])

            # 一致性比率 CR
            CR = CI / RI
        else:
            CR = torch.tensor(0)

        return {"W": W, "eigen_max": eigen_max, "CR": CR}