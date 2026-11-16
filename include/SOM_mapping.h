#ifndef SOM_MAPPING_H
#define SOM_MAPPING_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include "SOM.h"
using namespace std;

enum TexFilter {
    LINEAR,
    NEAREST
};

enum SizeFilter {
    MINIFICATION,
    MAGNIFICATON
};

struct LayerVoxels { // 擷取範圍中每一層距離值的體素
    vector<glm::vec3> pos; // 體素索引 
    vector<glm::vec3> color; // texture mapping後顏色值
};

glm::ivec3 findPosBMU(SomData &som_data, glm::vec3 input_data);
void twoStepMapping(LayerVoxels &layer_voxels, SomData &som_data); // 算出LayerVoxels的color

extern TexFilter tex_filter;
extern SizeFilter size_filter;
#endif