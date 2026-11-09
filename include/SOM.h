#define GLM_ENABLE_EXPERIMENTAL
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <limits>
#include <random>

using namespace std;

#define PI 3.14159265359

enum LatticesType {
    SQUARE, CUBE, CYLINDER, DONUT
};
// CUBE sequence: 按照 x(0~1), y(0~1), z(0~1)
//   4          back
// 0 3 1   left top right 
//   5          front
struct SomData {
    int resolution[3]; // neuron: width height faces
    vector<glm::vec3> texcoord; // r coordinate用不到 // 方便find BMU function的統一
    vector<glm::vec3> pos;

    LatticesType lattice_type; 

    SomData(int width, int height) {
        resolution[0] = width; 
        resolution[1] = height;
        resolution[2] = 1; // 預設為cube
        
        lattice_type = SQUARE;

        texcoord.resize(resolution[0] * resolution[1] * 1); // 固定是一個2D plane
        pos.resize(resolution[0] * resolution[1] * 6); // 固定大小為最大的CUBE
    };
    
    inline void setType(LatticesType type) {
        lattice_type = type;
        if(type == CUBE)
            resolution[2] = 6;
        else resolution[2] = 1;
    }

    int idx(int i, int j) const { 
        return j * resolution[0] + i; 
    }
    int idx(int i, int j, int k) const {
        return k * resolution[0] * resolution[1] + j * resolution[0] + i;
    }

    inline glm::vec3& operator()(int i, int j) {
        return texcoord[idx(i, j)];
    }
    inline glm::vec3& operator()(int i, int j, int k) {
        return pos[idx(i, j, k)];
    }
};

class SOM_c {
public:
    int resolution[3]; // 用來區分 3D lattices (donut) and texcoord 的訓練
    int iter_max = 120000;
    int iter_now;
    
    double learning_rate_init = 0.005;
    double learning_rate_now;

    double radius_init, radius_now;

    LatticesType lattice_type = CUBE; // 如果是訓練texture就一律設定為SQUARE
    
    // 只給texture用 或許可改?
    glm::ivec2 *anchor_points_idx = NULL; // 表index 
    glm::vec2 *anchor_points_texcoord = NULL; // 表texcoord

    void initiateParameters(SomData &som_data);
    void initiateLattice(vector<glm::vec3> &pos_data, SomData &som_data);
    void runSOM(vector<glm::vec3> &mesh_data, glm::vec3 input_vector);

private:
    glm::ivec3 findBMU(vector<glm::vec3> &mesh_data, glm::vec3 input_vector);
    double updateNeighbourhoodRadius();
    double calculateInfluence(glm::ivec3 BMU, glm::ivec3 now_point);
    double updateLearningRate();

    glm::ivec3 calculateCubeIndexPos(glm::ivec3 idx);
};

// 自己寫，因為新版glm的length2只限定float (為符合glsl)
inline int length2(glm::ivec3 data) { // for calculate distance
    return data.x * data.x + data.y * data.y + data.z * data.z;
};
inline float length2(glm::vec3 data) { // for calculate distance
    return data.x * data.x + data.y * data.y + data.z * data.z;
};