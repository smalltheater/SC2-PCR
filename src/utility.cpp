//
// Created by yunqi on 2022/7/6.
//
#include "utility.h"

void random_choice(int Num,int Range,int tensor[])
{
    std::vector<int> temp;
    for (int i=0;i<Range;i++)
    {
        temp.push_back(i);
    }
    std::random_shuffle(temp.begin(),temp.end());


    std::memcpy(tensor,&temp[0], sizeof(temp[0])*Num);


}







