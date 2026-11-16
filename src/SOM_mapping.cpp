#include "SOM_mapping.h"

glm::ivec3 findPosBMU(SomData &som_data, glm::vec3 input_data) {
    int n = 1;
    int *resolution = &som_data.resolution[0];
    double minimum_distance = std::numeric_limits<double>::max();
    glm::ivec3 winner(0, 0, 0);
    
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_real_distribution<> dis(0.0, 1.0);

    for(int k = 0; k < resolution[2]; k++) {
        for(int j = 0; j < resolution[1]; j++) {
            for(int i = 0; i < resolution[0]; i++) {
                glm::vec3 diff = som_data(i, j, k) - input_data;
                double distance = glm::dot(diff, diff);
                if (distance < minimum_distance)
                {
                    n = 1;
                    minimum_distance = distance;
                    winner = glm::ivec3(i, j, k);
                }
                else if (distance == minimum_distance)
                {
                    n++;
                    double random_value = dis(gen);

                    if(random_value < 1.0 / n)
                    {                    
                        winner = glm::ivec3(i, j, k);
                    }
                }
            }
        }
    }
    
    return winner;
}

void twoStepMapping(LayerVoxels &layer_voxels, SomData &som_data) {
    glm::ivec3 pos_BMU;
    for(auto voxel : layer_voxels.pos) {
        voxel += glm::vec3{0.5f, 0.5f, 0.5f};
        pos_BMU = findPosBMU(som_data, voxel);
    }
}