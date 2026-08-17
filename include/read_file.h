#ifndef READ_FILE_H
#define READ_FILE_H

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include "vertex.h"
using namespace std;

#define RESOLUTION 256 // 預設regular grid大小

inline vector<Vertex_c> vertices_tri; // 儲存STL/OBJ模型的原始三角片

enum SampleType
{
    INT, UNSIGNED_INT, CHAR, UNSIGNED_CHAR, FLOAT, UNSIGNED_SHORT, SHORT
};
enum EndianType
{
    LITTLE, BIG
};

///// 各種資料格式 /////
struct VecData {
    int resolution[2] = {0, 0};
    vector<glm::vec2> data;
    float max_speed = 0.0f;
    
    int idx(int i, int j) const { 
        return j * resolution[0] + i; 
    }

    inline glm::vec2& operator()(int i, int j) {
        return data[idx(i, j)];
    }
};

// 用regular grid存放scalar (intensity) 的voxel
struct VolumeData { 
    // 一個VolumeData也只有一組infdata跟rawdata 乾脆合併
    // struct InfData {} inf_data;
    glm::ivec3 resolution = {RESOLUTION, RESOLUTION, RESOLUTION}; // [width, height, depth]
    SampleType sample_type;
    glm::vec3 voxel_size;
    EndianType endian = LITTLE;
    glm::vec3 max, min;

    // struct RawData {} volume_data;
    int size;
    int intensity_counts[256] = {0};
    vector<unsigned char> voxel_data; // 真正存放體素資料

    int idx(int i, int j, int k) const {
        return k * resolution[0] * resolution[1] + j * resolution[0] + i;
    }
    
    inline unsigned char& operator()(int i, int j, int k) {
        return voxel_data[idx(i, j, k)];
    }
};

struct TxtData {
    int resolution;
    int dimension;
    vector<glm::vec3> weight;
    glm::vec3 max, min;
};


///// 讀各種檔案 /////
VecData ReadVecFile(const char *vec_filename);
bool ReadInfFile(const char *inf_filename, VolumeData &volume_data);
bool ReadRawFile(const char *raw_filename, VolumeData &volume_data);
VolumeData ReadVolumeFile(const char *inf_filename, const char *raw_filename);
TxtData ReadTxtFile(const char *txt_filename);
VolumeData ReadStlFile(const char *stl_filename);
VolumeData ReadObjFile(const char *stl_filename);

///// 一些中間函式 /////
void triangle2voxel(VolumeData &volume_data, glm::vec3 triangle_vertex[3]); // voxelization
template <typename T>
T ReadAndConvert(EndianType endian, std::ifstream& ifs);

///// 做資料後處裡 /////
vector<unsigned char> cumulativeDistributionEqualization(VolumeData &volume_data);
vector<unsigned char> adaptiveHistogramEqualization(VolumeData &volume_data, int side);
vector<unsigned char> CLAHE(VolumeData &volume_data, int block_edge, float threshold);

#endif