#ifndef TWE_STEP_MAPPING
#define TWE_STEP_MAPPING

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
using namespace std;

struct LayerVoxels { // 擷取範圍中每一層距離值的體素
    vector<glm::vec3> pos; // 體素索引 
    vector<glm::vec3> color; // texture mapping後顏色值
};

#endif