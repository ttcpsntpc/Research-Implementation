#ifndef DISTANCE_FIELD_3D_H
#define DISTANCE_FIELD_3D_H

#include <iostream>
#include <vector>
#include <tuple>
using namespace std;

/*
注意：
    模型自己用tuple定義，且希望直接能以現實位置定義，範圍固定從 0 to width/height，若width != height，函式內判斷須修改

說明：
    可以直接用這個類別計算模型的距離場，並分辨模型內外部，距離值為負代表在外部，為正則代表在內部

宣告：
    DistanceField3D name(width, height, depth);
計算距離場：
    name.compute(tuple<int, int, int> points);
    需要給他<vector>中tuple<int, int, int>的list，代表模型所在位置
可使用數值：
    getDistance(int x, int y, int z)可以得到該位置的距離值
    getMaxDistance()可以得到距離場中最大的距離值
*/

struct DistanceFieldData {
    int width, height, depth; 
    double max_distance;
    
    vector<double> voxel_distance;
    
    int idx(int i, int j, int k) const {
        return k * width * height + j * width + i;
    }
    
    inline double& operator()(int i, int j, int k) {
        return voxel_distance[idx(i, j, k)];
    }
};

enum VoxelState {
    INITIAL, DONE, CLOSE, FAR
};

struct Close_PQ { // for priority queue
    int i, j, k;
    double distance;
    
    // 為了讓 Priority Queue 變成 Min-Heap (由小排到大)，需重載大於運算子
    bool operator>(const Close_PQ& other) const {
        return distance > other.distance;
    }
};

struct Close_LL { // for linked list
    int i, j, k; // 存close_distance_list對應voxel_distance的index
    double distance; // 存CLOSE voxel的距離
    Close_LL *next = nullptr;
};

class DistanceField3D {
public:
    DistanceField3D() {};
    ~DistanceField3D() {};

    // main function to compute distance field
    DistanceFieldData compute_LL(const vector<std::tuple<int, int, int>>& points, int width, int height, int depth); // 可忽略，初步測試用
    DistanceFieldData compute_LL(const vector<unsigned char>& voxels, int width, int height, int depth); // 用linked list計算距離場
    DistanceFieldData compute_PQ(const vector<unsigned char>& voxels, int width, int height, int depth); // 用priority queue計算距離場

private:
    int width, height, depth; 
    double max_distance;
    
    vector<VoxelState> voxel_state;

    int offsets[6][3] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};

    bool isRangeValid(int i, int j, int k);
    void insertList(int i, int j, int k, double distance, Close_LL *&close_list_header); // for LL
    double computeDistance(DistanceFieldData &df_data, int i, int j, int k);
    bool isClose(int i, int j, int k);
    int idx(int i, int j, int k) { return i + j * width + k * width * height; }
    void determineInOut(vector<double> &voxel_distance);
};

#endif