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
    MAGNIFICATION
};

struct LayerVoxels { // 擷取範圍中每一層距離值的體素
    vector<glm::vec3> pos; // 體素索引 
    vector<glm::vec3> color; // texture mapping後顏色值
};

glm::ivec3 findPosBMU(SomData &som_data, glm::vec3 input_data);
bool inTriangle(glm::vec3 a, glm::vec3 b, glm::vec3 o, glm::vec3 projp);
glm::vec3 outerProjp(glm::vec3 a, glm::vec3 b, glm::vec3 o, glm::vec3 projp);
glm::vec3 getTriangleProjectionPoint(glm::vec3 a, glm::vec3 b, glm::vec3 o, glm::vec3 p, SomData &som_data);
bool isNodeIndexValid(SomData som_data, glm::ivec3 &A);
void twoStepMapping(LayerVoxels &layer_voxels, SomData &som_data, vector<vector<glm::vec3>> &image); // 算出LayerVoxels的color
glm::vec2 SOMMapping(glm::vec3 voxel, SomData &som_data);

extern glm::ivec3 offset[5]; // for BMU相鄰的三角形
extern TexFilter tex_filter;
extern SizeFilter size_filter;
#endif