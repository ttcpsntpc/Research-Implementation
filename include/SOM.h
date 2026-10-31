#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <limits>
#include <ramdon>

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
    vector<glm::vec3> texcoord; // r coordinate用不到 // 方便find BMU function的統一
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

    inline glm::vec3& operator()(int i, int j) {
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

    LatticesType lattice_type = CUBE; // 如果是訓練texture就一律設定為SQUARE
    
    void initiateParameters(SomData &som_data);
    void createLattices(vector<glm::ivec3> &pos_data, SomData &som_data);
    glm::ivec3 findBMU(vector<glm::vec3> data, glm::vec3 input_vector);
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

    // initial position
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

    // initial texcoord
    static random_device rd;
    static std::mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);

    for(int j = 0; j < resolution[1]; j++) {
        for(int i = 0; i < resolution[0]; i++) {
            som(i, j) = glm::vec3(dis(gen), dis(gen), 0.0f);
        }
    }
}

// 遍歷mesh找出BMU，距離最小的神經元index
glm::ivec3 SOM::findBMU(vector<glm::vec3> data, glm::vec3 input_vector)
{
    int k = 1;
    double minimum_distance = std::numeric_limits<double>::max();
    glm::ivec3 winner(0, 0, 0);
    
    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    for(int k = 0; k < resolution[2]; k++) {
        for(int j = 0; j < resolution[1]; j++) {
            for(int i = 0; i < resolution[0]; i++) {
                int idx = k * resolution[1] * resolution[0] + j * resolution[0] + i;
                double distance = glm::length(data[idx] - input_vector);
                
                if (distance < minimum_distance)
                {
                    k = 1;
                    minimum_distance = distance;
                    winner = glm::vec3(i, j, k);
                }
                else if (distance == minimum_distance)
                {
                    k++;
                    double random_value = dis(gen);

                    if(random_value < 1.0 / k)
                    {                    
                        winner = glm::vec3(i, j, k);
                    }
                }
            }
        }
    }

    return winner;
}

double UpdateNeighbourhoodRadius(double original_radius, int iteration_count, int total_iteration_count);
double UpdateLearningRate(double original_learning_rate, int iteration_count, int total_iteration_count);
double CalculateInfluence(glm::vec2 BMU, glm::vec2 now_point, int resolution, double neighbourhood_radius);
void CreateThread();
void RunSOM();


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