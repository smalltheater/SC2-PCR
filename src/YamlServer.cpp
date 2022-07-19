//
// Created by yunqi on 2022/7/5.
//

#include "YamlServer.h"
#include "utility.h"

YamlParam::YamlParam()
{
    YAML::Node config = YAML::LoadFile("/home/yunqi/Desktop/SC2-pcr/config/params.yaml");

    num_iterations   = config["num_iterations"].as<int>();

    ratio            = config["ratio"].as<float>();
    k1               = config["k1"].as<int>();
    k2               = config["k2"].as<int>();
    data_path        = config["data_path"].as<std::string>();
    data_path0       = config["data_path0"].as<std::string>();
    data_path1       = config["data_path1"].as<std::string>();
    inlier_threshold = config["inlier_threshold"].as<double>();
    d_thre           = config["d_thre"].as<double>();
    downsample       = config["downsample"].as<double>();
    num_node         = config["num_node"].as<int>();
    use_mutual       = config["use_mutual"].as<bool>();
    max_points       = config["max_points"].as<int>();
    nms_radius       = config["nms_radius"].as<double>();
    file_types       = config["file_types"].as<std::string>();

}
