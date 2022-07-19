//
// Created by yunqi on 2022/7/5.
//

#ifndef SC2PCR_UTILITY_H
#define SC2PCR_UTILITY_H
#include <iostream>
#include <string>
#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAGuard.h>
#include<complex>
#include<cstdlib>
#include<map>
#include<string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ia_ransac.h>//采样一致性
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>//
#include <pcl/filters/filter.h>//
#include <pcl/registration/icp.h>//icp配准
#include <pcl/visualization/pcl_visualizer.h>//可视化
#include "yaml-cpp/yaml.h"
#include "cnpy.h"
struct Numpy
{
    double *pt0,*pt1,*feat0,*feat1;
};

void random_choice(int Num,int Range,int tensor[]);






#endif //SC2PCR_UTILITY_H
