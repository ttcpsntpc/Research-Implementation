#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <limits>

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
    int resolution[3]; // width height slice
    vector<glm::vec2> texcoord;
    vector<glm::vec3> pos;

    SomData(int width, int height) {
        resolution[0] = width; 
        resolution[1] = height;
        resolution[2] = 1; // 預設為1

        texcoord.resize(resolution[0] * resolution[1] * resolution[2]); // 應該就不變了
        pos.resize(resolution[0] * resolution[1] * resolution[2]); // 若改成donut就會改變
    };

    int idx(int i, int j) const { 
        return j * resolution[0] + i; 
    }
    int idx(int i, int j, int k) const {
        return k * resolution[0] * resolution[1] + j * resolution[0] + i;
    }

    inline glm::vec2& operator()(int i, int j) {
        return texcoord[idx(i, j)];
    }
    inline glm::vec3& operator()(int i, int j, int k) {
        return pos[idx(i, j, k)];
    }
};

class SOM {
public:
    int resolution[3];
    int iter_max = 120000;
    int iter_now;
    
    double learning_rate_init = 0.005;
    double learning_rate_now;

    double radius_init, radius_now;

    LatticesType lattice_type = CUBE;
    
    void initiateParameters(SomData &som_data);
    void createLattices(vector<glm::ivec3> &pos_data, SomData &som_data);
};

// 切換訓練種類時要呼叫
void SOM::initiateParameters(SomData &som_data) {
    for(int i = 0; i < 3; i++)
        resolution[i] = som_data.resolution[i];
    
    learning_rate_now = learning_rate_init;
    radius_init = radius_now = max(resolution[0], resolution[1]);
}

// 根據要訓練的模型空間初始化lattices
void SOM::createLattices(vector<glm::ivec3> &pos_data, SomData &som_data) {
    glm::ivec3 min(numeric_limits<int>::max()), max(numeric_limits<int>::lowest()); // voxel index min/max // form a BB
    for(auto data : pos_data) {
        min = glm::min(min, data);
        max = glm::max(max, data);
    }
    glm::vec3 size = max - min; // 轉成float比較方便使用
    glm::vec3 center = glm::vec3(min.x + size.x / 2, min.y, min.z + size.z / 2);
    double x, y, z;

    if(lattice_type == SQUARE) { // 初始化於max z上的面
        double x_gap = size.x / (resolution[0] - 1);
        double z_gap = size.z / (resolution[1] - 1);

        for(int j = 0; j < resolution[1]; j++) {
            for(int i = 0; i < resolution[0]; i++) {
                som_data(i, j, 0) = glm::vec3(min.x + i * x_gap, max.y, min.z + j * z_gap);
            }
        }
    } 
    else if(lattice_type == CYLINDER) { // 高為y軸的圓柱
        double y_gap = size.y / (resolution[1] - 1);
        double radius = glm::max(size.x, glm::max(size.y, size.z)) / 2; // 最大分量除以2

        for(int j = 0; j < resolution[1]; j++) {
            for(int i = 0; i < resolution[0]; i++) {
                x = center.x + radius * cos(2 * PI * i / (resolution[0] - 1));
                y = min.y + j * y_gap;
                z = center.z + radius * sin(2 * PI * j / (resolution[0] - 1));

                som_data(i, j, 0) = glm::vec3(x, y, z);
            }
        }
    }
    else if(lattice_type == DONUT) { // 對稱軸為z軸的甜甜圈
        double R = (size.x + size.y) / 2 / 2;
        double r = R / 5;

        for(int j = 0; j < resolution[1]; j++) {
            for(int i = 0; i < resolution[0]; i++) {
                double phi = 2 * PI * i / (resolution[0] - 1);
                double theta = 2 * PI * j / (resolution[1] - 1);

                x = (R + r * cos(phi)) * cos(theta);
                y = (R + r * cos(phi)) * sin(theta);
                z = r * sin(phi);

                som_data(i, j, 0) = center + glm::vec3(x, y, z);
            }
        }
    } 
    else if(lattice_type == CUBE) {
        resolution[2] = 6;

        for(int j = 0; j < resolution[1]; j++) {
            for(int i = 0; i < resolution[0]; i++) {
                double ratio_i = i / (resolution[0] - 1);
                double ratio_j = j / (resolution[1] - 1);

                // y-z plane left
                x = 0;
                y = min.y + size.y * ratio_j;
                z = min.z + size.z * ratio_i;
                som_data(i, j, 0) = glm::vec3(x, y, z);

                // y-z plane right
                x = max.x;
                y = min.y + size.y * ratio_j;
                z = min.z + size.z * ratio_i;
                som_data(i, j, 1) = glm::vec3(x, y, z);

                // x-z plane down
                x = min.x + size.x * ratio_i;
                y = 0;
                z = min.z + size.z * ratio_j;
                som_data(i, j, 2) = glm::vec3(x, y, z);
                
                // x-z plane up
                x = min.x + size.x * ratio_i;
                y = max.y;
                z = min.z + size.z * ratio_j;
                som_data(i, j, 3) = glm::vec3(x, y, z);

                // x-y plne back
                x = min.x + size.x * ratio_i;
                y = min.y + size.y * ratio_j;
                z = 0;
                som_data(i, j, 4) = glm::vec3(x, y, z);

                // x-y plane front
                x = min.x + size.x * ratio_i;
                y = min.y + size.y * ratio_j;
                z = max.y;
                som_data(i, j, 5) = glm::vec3(x, y, z);
            }
        }
    } 
    else cout<<"invalid lattices type number: "<<lattice_type<<endl;
}

void CreateCylindricalMesh(int resolution, glm::vec3 max, glm::vec3 min);
glm::ivec2 FindBMU(glm::vec3 input_vector, int resolution);
double UpdateNeighbourhoodRadius(double original_radius, int iteration_count, int total_iteration_count);
double UpdateLearningRate(double original_learning_rate, int iteration_count, int total_iteration_count);
double CalculateInfluence(glm::vec2 BMU, glm::vec2 now_point, int resolution, double neighbourhood_radius);
void CreateThread();
void RunSOM();


// 初始化神經元(mesh)的位置為cylinder
void CreateCylindricalMesh(int resolution, glm::vec3 max, glm::vec3 min)
{
    // 網格的weight預設是以資料集的中心點，其xz平面畫圓，從y軸向上長的cylnder
    glm::vec3 length = max - min;
    float radius = (length.x + length.z) / 4;
    float heightStep = length.y / resolution;
    std::vector<Vertex_c> cylinder;
    for(float y = 0; y < resolution - 1; ++y){ // height - column 
        for(int i = 0; i < resolution; ++i){ // circle - row
            // connected between circles
            float angle = 2.0f * M_PI * i / resolution;
            float x = radius * cos(angle) + min.x + length.x / 2;
            float z = radius * sin(angle) + min.z + length.z / 2;
            cylindrical_mesh[y][i] = glm::vec3(x, y * heightStep, z);

            cylinder.push_back(Vertex_c{{x, y * heightStep, z}, {1.0f, 1.0f, 1.0f}, {}, {}});
            cylinder.push_back(Vertex_c{{x, (y + 1) * heightStep, z}, {1.0f, 1.0f, 1.0f}, {}, {}});

            // connected a complete circle
            cylinder.push_back(Vertex_c{{x, y * heightStep, z}, {1.0f, 1.0f, 1.0f}, {}, {}});
            angle = 2.0f * M_PI * (i + 1) / resolution;
            x = radius * cos(angle) + min.x + length.x / 2;
            z = radius * sin(angle) + min.z + length.z / 2;
            
            cylinder.push_back(Vertex_c{{x, y * heightStep, z}, {1.0f, 1.0f, 1.0f}, {}, {}});
        }
    }
    // 補畫最後一層少的圓
    for(int i = 0; i < resolution; ++i){    
        // connected the whole circle
        float angle = 2.0f * M_PI * i / resolution;
        float x = radius * cos(angle) + min.x + length.x / 2;
        float z = radius * sin(angle) + min.z + length.z / 2;
        cylindrical_mesh[resolution - 1][i] = glm::vec3(x, (resolution - 1) * heightStep, z);

        cylinder.push_back(Vertex_c{{x, (resolution - 1) * heightStep, z}, {1.0f, 1.0f, 1.0f}, {}, {}});
        angle = 2.0f * M_PI * (i + 1) / resolution;
        x = radius * cos(angle) + min.x + length.x / 2;
        z = radius * sin(angle) + min.z + length.z / 2;
        cylinder.push_back(Vertex_c{{x, (resolution - 1) * heightStep, z}, {1.0f, 1.0f, 1.0f}, {}, {}});
    }
    
    vertices.push_back(cylinder); // 2
    cout<<"vetices2 size: "<<vertices[2].size()<<endl;
}

// 遍歷mesh找出BMU，距離最小的神經元
glm::ivec2 FindBMU(glm::vec3 input_vector, int resolution)
{
    int k = 1;
    float minimum_distance = std::numeric_limits<float>::max();
    glm::ivec2 winner(0, 0);
    for(int i = 0; i < resolution; ++i)
    {
        for(int j = 0; j < resolution; ++j)
        {
            float distance = glm::length(cylindrical_mesh[i][j] - input_vector);
            if (distance < minimum_distance)
            {
                k = 1;
                minimum_distance = distance;
                winner.x = i;
                winner.y = j;
            }
            else if (distance == minimum_distance)
            {
                k++;
                // random number between 0 and 1
                // !!! for c++ 11
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(0.0, 1.0);
                double random_value = dis(gen);
                if(random_value < 1.0 / k)
                {
                    winner.x = i;
                    winner.y = j;
                }
            }
        }
    }
    return winner;
}

// 更新鄰域半徑
double UpdateNeighbourhoodRadius(double original_radius, int iteration_count, int total_iteration_count)
{
    return original_radius * exp( (double)-iteration_count * log(original_radius) / (double)total_iteration_count );
}

// 更新學習率
double UpdateLearningRate(double original_learning_rate, int iteration_count, int total_iteration_count)
{
    return original_learning_rate * exp( -(double)iteration_count / (double)total_iteration_count );
}

// 計算BMU對其他神經元的影響力，距離越遠影響力越小，超過鄰域半徑的神經元不受BMU影響
double CalculateInfluence(glm::vec2 BMU, glm::vec2 now_point, int resolution, double neighbourhood_radius)
{
    double distance = (BMU.x - now_point.x) * (BMU.x - now_point.x) + (BMU.y - now_point.y) * (BMU.y - now_point.y);
    if (distance > neighbourhood_radius * neighbourhood_radius)
    {
        //cout<<" pre_distance: "<<distance;
        // 配合cylinder的特性要考慮邊界情況，BMU和now_point在同一行但跨越邊界的情況
        if(BMU.y < now_point.y)
            distance = (resolution + BMU.y - now_point.y) * (resolution + BMU.y - now_point.y) + (BMU.x - now_point.x) * (BMU.x - now_point.x);
        else
            distance = (resolution + now_point.y - BMU.y) * (resolution + now_point.y - BMU.y) + (BMU.x - now_point.x) * (BMU.x - now_point.x);

        if(distance > neighbourhood_radius * neighbourhood_radius)
            return 0.0;
    }
    //cout<<" distance: "<<distance;
    return exp(-(distance / (2 * neighbourhood_radius * neighbourhood_radius)));
}

void CreateThread()
{
    if (t1.joinable())
    {
        t1.join();
    }
    t1 = std::thread(RunSOM);
    // t1.detach(); // if you want to run it in background
}

void RunSOM()
{
     // 1. pick a random input vector
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, rf.txt_data.weight.size() - 1);
    int random_index = dis(gen);    
    glm::vec3 input_vector = rf.txt_data.weight[random_index];

    // 2. find the best matching unit (BMU)
    glm::ivec2 BMU = FindBMU(input_vector, resolution);
    
    // 3. update the radius and learning rate
    neighbor_radius = UpdateNeighbourhoodRadius(original_radius, iteration_count, total_iteration_count);
    learning_rate = UpdateLearningRate(original_learning_rate, iteration_count, total_iteration_count);
    
    // cylindrical_mesh[BMU.x][BMU.y] += (float)learning_rate * (input_vector - cylindrical_mesh[BMU.x][BMU.y]); // 這行好像多此一舉
    for(int j = 0; j < resolution; ++j)
    {
        for(int k = 0; k < resolution; ++k)
        {
            glm::ivec2 now_point(j, k);
            double influence = CalculateInfluence(BMU, now_point, resolution, neighbor_radius);
            glm::ivec2 distance = BMU - now_point;
            // if( influence != 0.0){
            //     cout<<" cylinderical_mesh["<<j<<"]["<<k<<"] influence: "<<influence;
            //     cout<<"  BMU: "<<BMU.x<<", "<<BMU.y<<endl;
            // }
            cylindrical_mesh[j][k] += (float)influence * (float)learning_rate * (input_vector - cylindrical_mesh[j][k]);
        }
    }
    //cout<<endl;
}