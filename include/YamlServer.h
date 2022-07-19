//
// Created by yunqi on 2022/7/5.
//

#ifndef SC2PCR_YAMLSERVER_H
#define SC2PCR_YAMLSERVER_H
#include "utility.h"


class YamlParam
{
public:
    int num_iterations;
    std::string dataset;
    float ratio;
    int k1;
    int k2;
    std::string data_path;
    std::string data_path0;
    std::string data_path1;

    double inlier_threshold;
    double d_thre;
    double downsample;
    int num_node;
    bool use_mutual;
    int max_points;
    double nms_radius;
    std::string file_types;
public:
    YamlParam();
};

#endif //SC2PCR_YAMLSERVER_H
