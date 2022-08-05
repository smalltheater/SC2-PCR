#include "utility.h"
#include "YamlServer.h"
#include "SC2PCR.h"
#include "omp.h"
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/distances.h>
#include <pcl/kdtree/kdtree_flann.h>


void preprocess_point_cloud( pcl::PointCloud<pcl::PointXYZI>::Ptr cloud ,
                             pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh, float down_sample)
{
    pcl::VoxelGrid<pcl::PointXYZI> sor;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcd_down(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal> ());
    pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> ne;
    pcl::FPFHEstimationOMP<pcl::PointXYZI, pcl::Normal, pcl::FPFHSignature33> fpfhe;

    sor.setInputCloud(cloud);
    sor.setLeafSize(down_sample,down_sample,down_sample);
    sor.filter(*pcd_down);
    cout<<"Cloud filtered size is :"<<pcd_down->size()<<endl;

    ne.setInputCloud(pcd_down);
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI> ());
    ne.setSearchMethod (tree);
//    ne.setRadiusSearch(down_sample);
    ne.setKSearch(30);
    ne.compute(*normals);
//    cout<< "Cloud normal sizes is :"<< normals->size()<<endl;

    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZI> ());
    fpfhe.setSearchMethod(tree2);
    fpfhe.setInputCloud(pcd_down);
    fpfhe.setInputNormals(normals);
    fpfhe.setRadiusSearch(down_sample*2);
//    fpfhe.setKSearch(100);
    fpfhe.compute(*fpfh);

    cloud->clear();
    *cloud=*pcd_down;

//    cout<<"FPFH features sizes is :"<<fpfh->size()<<endl;


}

float caculateRMSE(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_source, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_target)
{
//    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_source(new pcl::PointCloud<pcl::PointXYZ>());
////    fromPCLPointCloud2(*cloud_source, *xyz_source);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_target(new pcl::PointCloud<pcl::PointXYZ>());
//    fromPCLPointCloud2(*cloud_target, *xyz_target);

    float rmse = 0.0f;

    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    tree->setInputCloud(cloud_target);

    for (auto point_i : *cloud_source)
    {
        // 去除无效的点
        if (!pcl_isfinite(point_i.x) || !pcl_isfinite(point_i.y) || !pcl_isfinite(point_i.z))
            continue;
        pcl::Indices nn_indices(1);
        std::vector<float> nn_distances(1);
        if (!tree->nearestKSearch(point_i, 1, nn_indices, nn_distances)) // K近邻搜索获取匹配点对
            continue;
        /*dist的计算方法之一
        size_t point_nn_i = nn_indices.front();
        float dist = squaredEuclideanDistance(point_i, xyz_target->points[point_nn_i]);
        */

        float dist = nn_distances[0]; // 获取最近邻对应点之间欧氏距离的平方
        rmse += dist;                 // 计算平方距离之和
    }
    rmse = std::sqrt(rmse / static_cast<float> (cloud_source->points.size())); // 计算均方根误差

    return rmse;
}



int main()
{
    YamlParam config;

    if(config.file_types=="npz") {

//
//        srand(0);
//
//        cnpy::npz_t arrt = cnpy::npz_load("/home/yunqi/Desktop/lt_slam/loop_pairs/npz/pair_0.npz");
//
//
//        cnpy::NpyArray xyz0 = arrt["xyz0"];
//        cnpy::NpyArray xyz1 = arrt["xyz1"];
//
//        cnpy::NpyArray features0 = arrt["features0"];
//        cnpy::NpyArray features1 = arrt["features1"];
//
//
//        double *feat0 = features0.data<double>();
//        double *feat1 = features1.data<double>();
//
//
//        double *pt0 = xyz0.data<double>();
//        double *pt1 = xyz1.data<double>();
//
//
//        auto T_pt0 = torch::from_blob(pt0, {int(xyz0.shape[0]), int(xyz0.shape[1])}, torch::kFloat64).unsqueeze(0).to(
//                torch::kFloat32);
//        auto T_pt1 = torch::from_blob(pt1, {int(xyz1.shape[0]), int(xyz1.shape[1])}, torch::kFloat64).unsqueeze(0).to(
//                torch::kFloat32);
//
//
//        auto T_feat0 = torch::from_blob(feat0, {int(features0.shape[0]), int(features0.shape[1])}, torch::kFloat64).to(
//                torch::kFloat32);
//        auto T_feat1 = torch::from_blob(feat1, {int(features1.shape[0]), int(features1.shape[1])}, torch::kFloat64).to(
//                torch::kFloat32);
//
//        auto input0 = torch::div(T_feat0, torch::norm(T_feat0, 2, 1, true) + 1e-6).to(torch::kFloat32).unsqueeze(0);
//        auto input1 = torch::div(T_feat1, torch::norm(T_feat1, 2, 1, true) + 1e-6).to(torch::kFloat32).unsqueeze(0);
//
//        auto input0_cuda = input0.to(torch::kCUDA);
//        auto input1_cuda = input1.to(torch::kCUDA);
//
//        auto T_pt0_cuda = T_pt0.to(torch::kCUDA);
//        auto T_pt1_cuda = T_pt1.to(torch::kCUDA);
//
//
//        SC2PCR matcher;
//
//        auto pair_pts = matcher.match_pair(T_pt0_cuda, T_pt1_cuda, input0_cuda, input1_cuda);
//        auto res = matcher.SC2_PCR(pair_pts.first, pair_pts.second);
//
//
//        std::cout << res << std::endl;
    }
    else if(config.file_types=="pcd")
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud0_raw(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud1_raw(new pcl::PointCloud<pcl::PointXYZI>);


        if(pcl::io::loadPCDFile<pcl::PointXYZI>(config.data_path0,*cloud0)==-1 || pcl::io::loadPCDFile<pcl::PointXYZI>(config.data_path1,*cloud1)==-1)
        {
            PCL_ERROR("Couldn't read file for pcd.");
            return -1;
        }


        pcl::io::loadPCDFile<pcl::PointXYZI>(config.data_path0,*cloud0);
        pcl::io::loadPCDFile<pcl::PointXYZI>(config.data_path1,*cloud1);
        pcl::copyPointCloud(*cloud0,*cloud0_raw);
        pcl::copyPointCloud(*cloud1,*cloud1_raw);
        std::cout<<"Load cloud 0 with "<<cloud0->points.size()<<" points from file"<<std::endl;
        std::cout<<"Load cloud 1 with "<<cloud1->points.size()<<" points from file"<<std::endl;

        TicToc t(true);


        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh0(new pcl::PointCloud<pcl::FPFHSignature33> ());
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh1(new pcl::PointCloud<pcl::FPFHSignature33> ());

        preprocess_point_cloud(cloud0,fpfh0,0.3f);
        preprocess_point_cloud(cloud1,fpfh1,0.3f);
        t.toc("PrePROCESSED");



        float pc0[cloud0->size()*3];
        float pc1[cloud1->size()*3];
        float feat0[fpfh0->size()*33];
        float feat1[fpfh1->size()*33];

t.tic();

torch::set_num_interop_threads(8);
        omp_set_num_threads(8);


            for(int i=0;i<cloud0->size();i++)
            {
                pc0[i*3]=cloud0->points[i].x;
                pc0[i*3+1]=cloud0->points[i].y;
                pc0[i*3+2]=cloud0->points[i].z;
                for(int j=0;j<33;j++)
                {
                    feat0[i*33+j]=fpfh0->points[i].histogram[j];
                }

            }



        t.toc("omp");

        for(int i=0;i<cloud1->size();i++)
        {

            pc1[i*3]=cloud1->points[i].x;
            pc1[i*3+1]=cloud1->points[i].y;
            pc1[i*3+2]=cloud1->points[i].z;
            for(int j=0;j<33;j++)
            {
                feat1[i*33+j]=fpfh1->points[i].histogram[j];
            }
        }


        torch::DeviceType device_type;
        device_type = torch::kCUDA;
        torch::Device device(device_type);

        torch::Tensor T_pt0 = torch::from_blob(pc0, {int(cloud0->size()), int(3)},torch::kFloat32).unsqueeze(0).to(device);
        torch::Tensor T_pt1 = torch::from_blob(pc1, {int(cloud1->size()), int(3)}, torch::kFloat32).unsqueeze(0).to(device);






        torch::Tensor  T_feat0 = torch::from_blob(feat0, {int(fpfh0->size()), int(33)}, torch::kFloat32).to(device);
        torch::Tensor  T_feat1 = torch::from_blob(feat1, {int(fpfh1->size()), int(33)}, torch::kFloat32).to(device);

        auto input0 = torch::div(T_feat0, torch::norm(T_feat0, 2, 1, true) + 1e-6).to(torch::kFloat32).unsqueeze(0);
        auto input1 = torch::div(T_feat1, torch::norm(T_feat1, 2, 1, true) + 1e-6).to(torch::kFloat32).unsqueeze(0);


        SC2PCR matcher;

        auto pair_pts = matcher.match_pair(T_pt0, T_pt1, input0, input1);

        auto res = matcher.SC2_PCR(pair_pts.first, pair_pts.second);

        t.toc("Finished");


        Eigen::Matrix4f T=Eigen::Matrix4f::Identity();
        T(0,0)=res[0][0][0].item().toFloat();
        T(0,1)=res[0][0][1].item().toFloat();
        T(0,2)=res[0][0][2].item().toFloat();
        T(0,3)=res[0][0][3].item().toFloat();
        T(1,0)=res[0][1][0].item().toFloat();
        T(1,1)=res[0][1][1].item().toFloat();
        T(1,2)=res[0][1][2].item().toFloat();
        T(1,3)=res[0][1][3].item().toFloat();
        T(2,0)=res[0][2][0].item().toFloat();
        T(2,1)=res[0][2][1].item().toFloat();
        T(2,2)=res[0][2][2].item().toFloat();
        T(2,3)=res[0][2][3].item().toFloat();

        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("Original PointCloud"));
        viewer->setBackgroundColor (255, 255, 255);

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color1 (cloud0_raw, 0, 2, 255);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color2 (cloud1_raw, 255, 0, 0);
        viewer->addPointCloud<pcl::PointXYZI> (cloud0_raw, color1, "target");
        viewer->addPointCloud<pcl::PointXYZI> (cloud1_raw, color2, "final");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
        viewer->addCoordinateSystem (0.05);
        viewer->initCameraParameters ();



        pcl::transformPointCloud(*cloud0_raw,*cloud0_raw,T.matrix());
        pcl::transformPointCloud(*cloud0,*cloud0,T.matrix());

        pcl::visualization::PCLVisualizer::Ptr viewer2 (new pcl::visualization::PCLVisualizer ("PointCloud with registration"));
        viewer2->setBackgroundColor (255, 255, 255);

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color3 (cloud0_raw, 0, 2, 255);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color4 (cloud1_raw, 255, 0, 0);
        viewer2->addPointCloud<pcl::PointXYZI> (cloud0_raw, color3, "target");
        viewer2->addPointCloud<pcl::PointXYZI> (cloud1_raw, color4, "final");
        viewer2->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
        viewer2->addCoordinateSystem (0.05);
        viewer2->initCameraParameters ();

        while (!viewer2->wasStopped ()&&!viewer->wasStopped())
        {
            viewer->spinOnce(100);
            viewer2->spinOnce (100);

        }


        pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
        icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align pointclouds
        icp.setInputSource(cloud0);
        icp.setInputTarget(cloud1);

        pcl::PointCloud<pcl::PointXYZI>::Ptr unused_result(new pcl::PointCloud<pcl::PointXYZI>());
        icp.align(*unused_result);

        std::cout<<"The fitness score is: "<<icp.getFitnessScore()<<endl;

        t.tic();
        std::cout<<caculateRMSE(cloud0_raw,cloud1_raw)<<std::endl;
        t.toc("RMSE");
        std::cout<<caculateRMSE(cloud0,cloud1)<<std::endl;


    }



}