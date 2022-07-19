//
// Created by yunqi on 2022/7/5.
//

#ifndef DCGAN_SC2PCR_H
#define DCGAN_SC2PCR_H
#include "utility.h"
#include "YamlServer.h"
#include "tictoc.h"


class SC2PCR : public YamlParam
{
public:
//    SC2PCR();
    std::pair<torch::Tensor ,torch::Tensor> match_pair(torch::Tensor &src_keypts, torch::Tensor &tgt_keypts, torch::Tensor &src_features, torch::Tensor &tgt_features);
    torch::Tensor SC2_PCR(torch::Tensor &src_keypts, torch::Tensor &tgt_keypts);
    torch::Tensor cal_leading_eigenvector(torch::Tensor M);
    torch::Tensor pick_seeds(torch::Tensor &dists, torch::Tensor &score ,double nms_radius, int max_num);
    torch::Tensor cal_seed_trans(torch::Tensor &seeds, torch::Tensor &SC2_measure, torch::Tensor &src_keypts, torch::Tensor &tgt_keypts);
    torch::Tensor rigid_transform_3d(torch::Tensor &A, torch::Tensor &B, torch::Tensor &weights);
    torch::Tensor post_refinement(torch::Tensor &initial_trans, torch::Tensor &src_keypts,torch::Tensor &tgt_keypts, int it_num);
    torch::Tensor transform(torch::Tensor &pts, torch::Tensor &trans);
};

#endif //DCGAN_SC2PCR_H
