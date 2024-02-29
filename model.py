import sys
from typing import List, Dict

import torch


class APH(torch.nn.Module):
    def __init__(self, **kwargs):
        super(APH, self).__init__()
        assert "device" in kwargs
        self.device = kwargs["device"]

    def forward(self, *args, **kwargs) -> Dict:
        """
        :param args:  层级列表 从最大层级开始  e.g. [metrics_one, metrics_two, assessment_sheet(最后的评价问卷表)]
        :param kwargs:  模糊综合判断矩阵 e.g. {metrics_one: List[torch.Tensor], metrics_two: List[torch.Tensor],
        assessment_sheet: List[torch.Tensor]}

        :return:

        """
        args = args[0]

        result_dict = {args[0]: [self.consistency_checks(z) for z in kwargs[args[0]]]}

        for metrics_index in range(1, len(args)-1):
            metrics = args[metrics_index]
            matrix_list = kwargs[metrics]
            result_dict[metrics] = []
            for f in result_dict[args[metrics_index-1]]:
                fw = f["combine_W"] if "combine_W" in f.keys() else f["W"]
                current_list = matrix_list[:len(fw)]
                matrix_list = matrix_list[len(fw):]
                for matrix_index in range(len(current_list)):
                    matrix = current_list[matrix_index]
                    matrix_result = self.consistency_checks(matrix)
                    matrix_result["combine_W"] = fw[matrix_index] * matrix_result["W"]
                    result_dict[args[metrics_index]].append(matrix_result)

        S = torch.mm(result_dict[args[-2]][0]["W"].T, kwargs[args[-1]][0])

        for matrix_index in range(1, len(kwargs[args[-1]])):
            matrix = kwargs[args[-1]][matrix_index]
            S = torch.cat((S, torch.mm(result_dict[args[-2]][matrix_index]["W"].T, matrix)), dim=0)

        result_dict[args[-1]] = [torch.mm(result_dict[args[0]][0]["W"].T, S)]


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
            CR = torch.tensor(0.)

        return {"W": W, "eigen_max": eigen_max, "CR": CR}
