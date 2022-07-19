//
// Created by yunqi on 2022/7/5.
//

#include "SC2PCR.h"
#include "utility.h"

using namespace std;
using namespace torch::indexing;

std::pair<torch::Tensor, torch::Tensor>
SC2PCR::match_pair(torch::Tensor &src_keypts, torch::Tensor &tgt_keypts, torch::Tensor &src_features,
                   torch::Tensor &tgt_features)
{

    int N_src = src_features.sizes()[1];
    int N_tgt = tgt_features.sizes()[1];



    torch::Tensor src_sel_idx;
    torch::Tensor tgt_sel_idx;



    if(num_node==0)
    {
         src_sel_idx=torch::arange(N_src,torch::kInt32).to(torch::kLong).to(src_keypts.device());
         tgt_sel_idx=torch::arange(N_tgt,torch::kInt32).to(torch::kLong).to(tgt_keypts.device());

    } else
    {

        int src_array[num_node];
        int tgt_array[num_node];
        random_choice(num_node,N_src,src_array);
        random_choice(num_node,N_tgt,tgt_array);
        src_sel_idx=torch::from_blob(src_array,num_node,torch::kInt32).to(torch::kLong).to(src_keypts.device());
        tgt_sel_idx=torch::from_blob(tgt_array,num_node,torch::kInt32).to(torch::kLong).to(tgt_keypts.device());

    }


    auto src_desc=src_features.index_select(1,src_sel_idx);
    auto tgt_desc=tgt_features.index_select(1,tgt_sel_idx);

    auto src_keypts_filtered=src_keypts.index_select(1,src_sel_idx).to(torch::kCUDA);
    auto tgt_keypts_filtered=tgt_keypts.index_select(1,tgt_sel_idx).to(torch::kCUDA);


   torch::Tensor distance=torch::sqrt(2- 2*torch::mm(src_desc[0],tgt_desc[0].transpose(1,0) )+1e-6 ).to(torch::kCUDA);



    auto distance_us=distance.unsqueeze(0);

    auto source_idx=torch::argmin(distance_us[0],1);

    auto corr=torch::stack({torch::arange(source_idx.sizes()[0],torch::kInt32).to(torch::kCUDA),source_idx},0);


    auto src_keypts_corr = src_keypts_filtered.index_select(1, corr[0]);

    auto tgt_keypts_corr= tgt_keypts_filtered.index_select(1,corr[1]);


    return {src_keypts_corr,tgt_keypts_corr};

}

torch::Tensor SC2PCR::SC2_PCR(torch::Tensor &src_keypts, torch::Tensor &tgt_keypts)
{


    int num_corr=src_keypts.sizes()[1];

    auto src_dist=torch::norm(src_keypts.unsqueeze(2)-src_keypts.unsqueeze(1),2,-1).to(torch::kFloat32).to(torch::kCUDA);
    auto tgt_dist=torch::norm(tgt_keypts.unsqueeze(2)-tgt_keypts.unsqueeze(1),2,-1).to(torch::kFloat32).to(torch::kCUDA);




    auto cross_dist=torch::abs(src_dist-tgt_dist);

/*
 * compute first order measure
 */

    float SC_dist_thre=d_thre;
    auto SC_measure= torch::clamp(1.0-torch::square(cross_dist)/(SC_dist_thre*SC_dist_thre),0).to(torch::kCUDA);
//    cout<<setprecision(10)<<SC_measure[0][0][1]<<endl;
//    cout<<SC_measure[0][252][4]<<endl;
    torch::Tensor mask1=torch::ones({SC_measure.sizes()}).to(torch::kFloat32).to(torch::kCUDA)*d_thre;

    auto hard_SC_measure=torch::ceil(torch::clamp(mask1-cross_dist,0,1)).to(torch::kFloat32).to(torch::kCUDA);
//    cout<<hard_SC_measure[0][4][252]<<endl;


/*
 * select reliable seed correspondences
 */
    auto confidence = cal_leading_eigenvector(SC_measure);




    auto seeds= pick_seeds(src_dist,confidence,nms_radius,int(num_corr*ratio));
/*
# compute second order measure
*/
    float SC2_dist_thre=d_thre/2;

    torch::Tensor mask2=torch::ones({SC_measure.sizes()},torch::kFloat32).to(torch::kCUDA)*SC2_dist_thre;


    auto hard_SC_measure_tight=torch::ceil(torch::clamp(mask2-cross_dist,0,1));
    auto seed_hard_SC_measure=hard_SC_measure.gather(1,seeds.unsqueeze(2).expand({-1,-1,num_corr}));


    auto seed_hard_SC_measure_tight=hard_SC_measure_tight.gather(1,seeds.unsqueeze(2).expand({-1,-1,num_corr}));
//    cout<<seed_hard_SC_measure_tight[0][58][5]<<endl;

//Todo : wrong value for sc2_measure
    auto SC2_measure = torch::matmul(seed_hard_SC_measure_tight, hard_SC_measure_tight) * seed_hard_SC_measure;

    /*
     * compute the seed-wise transformations and select the best one
     */
    auto final_trans= cal_seed_trans(seeds,SC2_measure,src_keypts,tgt_keypts);
//    cout<<final_trans<<endl;

    /*
     * refine the result by recomputing the transformation over the whole set
     */

    final_trans = post_refinement(final_trans,src_keypts,tgt_keypts,20);
//    cout<<final_trans<<endl;

    return final_trans;
}

torch::Tensor SC2PCR::cal_leading_eigenvector(torch::Tensor M)
{

    torch::Tensor leading_eig = torch::ones({M.sizes()[0],M.sizes()[1],1}).to(torch::kCUDA);


    auto leading_eig_last = leading_eig;

    for(int i=0;i<num_iterations;i++)
    {
        leading_eig=torch::bmm(M,leading_eig);
//        cout<<i<<endl;


        leading_eig = leading_eig / (torch::norm(leading_eig, 2, 1,true) + 1e-6);
        if(torch::allclose(leading_eig,leading_eig_last))
            break;
        leading_eig_last=leading_eig;


    }
    leading_eig=leading_eig.squeeze(-1);

    auto result=leading_eig.to(torch::kCUDA);
    return result;

}

torch::Tensor SC2PCR::pick_seeds(torch::Tensor &dists, torch::Tensor &scores, double nms_radius, int max_num)
{
auto scores_c=scores.to(torch::kCPU);
auto dists_c =dists.to(torch::kCPU);

auto score_relation=scores_c.t()>=scores_c;


score_relation=score_relation.to(torch::kBool) | (dists_c[0] >= nms_radius).to(torch::kBool);



auto is_local_max = get<0>(score_relation.min(-1));
//cout<<is_local_max.sizes()<<endl;
auto score_local_max = scores_c * is_local_max;

//cout<<score_local_max.sizes()<<endl;

auto sorted_score=torch::argsort(score_local_max,1,true);

//cout<<sorted_score.sizes()<<endl;

auto return_idx = sorted_score.index_select(1,torch::arange(0,max_num)).to(torch::kCUDA);



return return_idx;
}

torch::Tensor SC2PCR::cal_seed_trans(torch::Tensor &seeds, torch::Tensor &SC2_measure, torch::Tensor &src_keypts,
                                     torch::Tensor &tgt_keypts)
{


    int bs=SC2_measure.sizes()[0];
    int num_corr=SC2_measure.sizes()[1];
    int num_channels=SC2_measure.sizes()[2];

    auto src_keypts_cuda=src_keypts.to(torch::kCUDA);
    auto tgt_keypts_cuda=tgt_keypts.to(torch::kCUDA);



    if(k1>num_channels)
    {
        k1=4;
        k2=4;
    }
    /*
     The first stage consensus set sampling
     Finding the k1 nearest neighbors around each seed
    */

    auto sorted_score=torch::argsort(SC2_measure,2, true);
    auto knn_idx=sorted_score.index_select(2,torch::arange(0,k1).to(torch::kCUDA));

    auto sorted_value=torch::sort(SC2_measure,2, true);

    auto idx_tmp=knn_idx.contiguous().view({bs,-1}).unsqueeze(2).expand({-1,-1,3}).to(torch::kCUDA);
    /*
     *construct the local SC2 measure of each consensus subset obtained in the first stage.
     */



    auto src_knn = src_keypts_cuda.gather(1, idx_tmp).view({bs, -1, k1, 3});  //[bs, num_seeds, k, 3]
    auto tgt_knn = tgt_keypts_cuda.gather(1, idx_tmp).view({bs, -1, k1, 3});

    auto src_dist = torch::pow(torch::pow(src_knn.unsqueeze(3)-src_knn.unsqueeze(2),2).sum(-1),0.5);
    auto tgt_dist = torch::pow(torch::pow(tgt_knn.unsqueeze(3)-tgt_knn.unsqueeze(2),2).sum(-1),0.5);


    auto cross_dist          = torch::abs(src_dist-tgt_dist);

    torch::Tensor mask               = torch::ones({cross_dist.sizes()},torch::kFloat32).to(torch::kCUDA)*d_thre;

    auto local_hard_SC_measure= torch::ceil(torch::clamp(mask-cross_dist,0,1));

    auto local_SC2_measure    = torch::matmul(local_hard_SC_measure.index_select(2,torch::tensor({1}).to(torch::kCUDA)),local_hard_SC_measure);

    /*
     * perform second stage consensus set sampling
     */

    sorted_score=torch::argsort(local_SC2_measure,3, true);
    auto knn_idx_fine = sorted_score.index_select(3,torch::arange(0,k2).to(torch::kCUDA));

    /*
     * construct the soft SC2 matrix of the consensus set
     */

    int num           =      knn_idx_fine.sizes()[1];
    knn_idx_fine      =      knn_idx_fine.contiguous().view({bs, num, -1}).unsqueeze(3);
    knn_idx_fine      =      knn_idx_fine.expand({-1,-1,-1,3});
    auto src_knn_fine =      src_knn.gather(2,knn_idx_fine).view({1,-1,k2,3});
    auto tgt_knn_fine =      tgt_knn.gather(2,knn_idx_fine).view({1,-1,k2,3});

    src_dist          =     torch::pow(torch::pow(src_knn_fine.unsqueeze(3) - src_knn_fine.unsqueeze(2) , 2).sum(-1) , 0.5);
    tgt_dist          =     torch::pow(torch::pow(tgt_knn_fine.unsqueeze(3) - tgt_knn_fine.unsqueeze(2) , 2).sum(-1) , 0.5);

    cross_dist        =     torch::abs(src_dist-tgt_dist);
    auto local_hard_measure = (cross_dist<d_thre*2).to(torch::kFloat32);
    local_SC2_measure=torch::matmul(local_hard_measure,local_hard_measure)/k2;

    auto local_SC_measure = torch::clamp(1 - torch::square(cross_dist)  / (d_thre * d_thre), 0);
    local_SC2_measure=local_SC_measure;
    local_SC2_measure=local_SC2_measure.view({-1,k2,k2});

   /*
    * Power iteratation to get the inlier probability
    */

    local_SC2_measure.index_select(1, torch::arange(0,local_SC2_measure.sizes()[1]).to(torch::kCUDA)).index_select(2,torch::arange(0,local_SC2_measure.sizes()[1]).to(torch::kCUDA))=0;

    auto total_weight= cal_leading_eigenvector(local_SC2_measure);
    total_weight=total_weight.view({1,-1,k2});
    total_weight =total_weight / (torch::sum(total_weight,-1, true)+1e-6);

    /*
     * calculate the transformation by weighted least-squares for each subsets in parallel
     */

    total_weight = total_weight.view({-1,k2});
    src_knn=src_knn_fine;
    tgt_knn=tgt_knn_fine;
    src_knn=src_knn.view({-1,k2,3});
    tgt_knn=tgt_knn.view({-1,k2,3});
    /*
     * compute the rigid transformation for each seed by the weighted SVD
     */

    auto seedwise_trans = rigid_transform_3d(src_knn,tgt_knn,total_weight);
    seedwise_trans=seedwise_trans.view({bs,-1,4,4});

    /*
     * calculate the inlier number for each hypothesis, and find the best transformation for each
     */

    auto pred_position = torch::einsum("bsnm,bmk->bsnk",
       torch::TensorList {seedwise_trans.index({Slice(None),Slice(None),Slice(0,3),Slice(0,3)})
                          ,src_keypts_cuda.permute({0,2,1})})+seedwise_trans.index({Slice(None),Slice(None),Slice(0,3),Slice(3,4)});

    /*
     * calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
     */

    pred_position = pred_position.permute({0,1,3,2});
    auto L2_dis   = torch::norm(pred_position-tgt_keypts_cuda.unsqueeze(1),2,-1);

    auto mask2 = torch::ones_like(L2_dis,torch::kFloat32).to(torch::kCUDA);
    auto seedwise_fitness = torch::sum((L2_dis<mask2).to(torch::kFloat32),-1);


    auto batch_best_guess = seedwise_fitness.argmax(1);
    auto final_trans = seedwise_trans.gather(1,batch_best_guess.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand({-1, -1, 4, 4})).squeeze(1);

    return final_trans;
}

torch::Tensor SC2PCR::rigid_transform_3d(torch::Tensor &A, torch::Tensor &B, torch::Tensor &weights)
{
    int bs=A.sizes()[0];

    auto temp= weights<0;
    weights.index({temp})=0;
//    cout<<"It works 0"<<endl;
    //find mean of point cloud

    auto centroid_A = torch::sum(A * weights.unsqueeze(2), 1, true) / (torch::sum(weights, 1, true).unsqueeze(2) + 1e-6);
    auto centroid_B = torch::sum(B * weights.unsqueeze(2), 1, true) / (torch::sum(weights, 1, true).unsqueeze(2) + 1e-6);
//    cout<<"It works 1"<<endl;
    // Subtract mean
    auto Am =A-centroid_A;
    auto Bm =B-centroid_B;
//    cout<<"It works 2"<<endl;

    // construct weight covariance matrix
    auto Weight=torch::diag_embed(weights);
    auto H = torch::matmul( torch::matmul(Am.permute({0,2,1}),Weight),Bm);
//    cout<<"It works 3"<<endl;
   //find rotation
    auto SVD=torch::svd(H.to(torch::kCPU));
    auto U=get<0>(SVD).to(torch::kCUDA);
    auto S=get<1>(SVD).to(torch::kCUDA);
    auto Vt=get<2>(SVD).to(torch::kCUDA);
//    cout<<"It works 4"<<endl;

    auto delta_UV = torch::det(torch::matmul(Vt,U.permute({0,2,1})));
//    cout<<delta_UV.sizes()<<endl;
//    cout<<setprecision(15)<<delta_UV[2]<<" "<<delta_UV[3]<<" "<<delta_UV[5]<<endl;


    auto eye = torch::eye(3).unsqueeze(0).repeat({bs, 1, 1}).to(torch::kCUDA);

    eye.index({"...",Slice(2),Slice(2)})=delta_UV.unsqueeze(1).unsqueeze(1);

    auto R= torch::matmul( torch::matmul(Vt,eye), U.permute({0,2,1}));
    auto t= centroid_B.permute({0,2,1})-torch::matmul(R,centroid_A.permute({0,2,1}));

    auto trans=torch::eye(4).repeat({R.sizes()[0],1,1}).to(R.device());

    trans.index({"...",Slice(0,3),Slice(0,3)})=R;
    trans.index({"...",Slice(0,3),Slice(3)})=t.view({-1,3,1});

    return trans;

}

torch::Tensor SC2PCR::post_refinement(torch::Tensor &initial_trans, torch::Tensor &src_keypts,
                                      torch::Tensor &tgt_keypts, int it_num)
    {

    double inlier_threshold_p=1.2;
    int previous_inlier_num = 0;


    while(true) {

        auto warped_src_keypts = transform(src_keypts, initial_trans);
        auto L2_dis = torch::norm(warped_src_keypts - tgt_keypts.to(torch::kCUDA), 2, -1);
        torch::Tensor pred_inlier = (L2_dis < inlier_threshold_p)[0].unsqueeze(0).unsqueeze(2).expand({-1,-1,3});
        torch::Tensor pred_inlier_b = (L2_dis < inlier_threshold_p)[0].unsqueeze(0);




        auto inlier_num = torch::sum(pred_inlier);
        it_num--;
        if (it_num < 0)
            break;
        else if (abs(int(inlier_num.item<long>()) - previous_inlier_num) < 1)
            break;
        else {
            previous_inlier_num = int(inlier_num.item<long>());
        }

        auto A = src_keypts.index( {pred_inlier}).view({1,-1,3}).to(torch::kCUDA);

//
        auto B = tgt_keypts.masked_select( pred_inlier).view({1,-1,3}).to(torch::kCUDA);
        auto weights =(1 / (1 + torch::square(L2_dis / inlier_threshold))).masked_select(pred_inlier_b).unsqueeze(0);

        cout<<weights.sizes()<<endl;
        initial_trans = rigid_transform_3d(A, B,weights);

    }

    return initial_trans;






    }

torch::Tensor SC2PCR::transform(torch::Tensor &pts, torch::Tensor &trans)
{
    auto pts_cuda=pts.to(torch::kCUDA);
//    cout<<trans[0]<<endl;
    auto trans_pts=torch::matmul(trans.index({Slice(None),Slice(0,3),Slice(0,3)}),
                                 pts_cuda.permute({0,2,1}))+trans.index({Slice(None),Slice(0,3),Slice(3,4)});
    return trans_pts.permute({0,2,1});
}





