#include <omp.h>
#include "SOM.h"


// 切換訓練種類時要呼叫
void SOM_c::initiateParameters(SomData &som_data) {
    lattice_type = som_data.lattice_type;
    for(int i = 0; i < 3; i++)
        resolution[i] = som_data.resolution[i];
    
    iter_now = 0;
    learning_rate_now = learning_rate_init;
    radius_init = radius_now = max(resolution[0], resolution[1]);

    if(anchor_points_idx == NULL) { // 只有第一次要初始化 之後可透過使用者修改
        anchor_points_idx = new glm::ivec2[5];
        anchor_points_idx[0] = glm::ivec2(0, 0);
        anchor_points_idx[1] = glm::ivec2(resolution[0] - 1, 0);
        anchor_points_idx[2] = glm::ivec2(resolution[0] - 1, resolution[1] - 1);
        anchor_points_idx[3] = glm::ivec2(0, resolution[1] - 1);
        anchor_points_idx[4] = glm::ivec2((resolution[0] - 1) / 2, (resolution[1] - 1) / 2);

        anchor_points_texcoord = new glm::vec2[5];
        for(int j = 0; j < 5; j++) {
            anchor_points_texcoord[j] = glm::ivec2(anchor_points_idx[j].x / (resolution[0] - 1), anchor_points_idx[j].y / (resolution[1] - 1));
        }
    }
}

// 根據要訓練的模型空間初始化lattice
void SOM_c::initiateLattice(vector<glm::vec3> &pos_data, SomData &som_data) {
    glm::vec3 min(numeric_limits<float>::max()), max(numeric_limits<float>::lowest()); // voxel index min/max // form a BB
    for(auto data : pos_data) {
        min = glm::min(min, data);
        max = glm::max(max, data);
    }
    glm::vec3 size = max - min; // 轉成float比較方便使用
    glm::vec3 center = glm::vec3(min.x + size.x / 2, min.y + size.y / 2, min.z + size.z / 2);
    double x, y, z;
    double ratio_i, ratio_j;
    min -= size * 0.2f;
    max += size * 0.2f;
    size = max - min;

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
        double radius = glm::max(size.x, size.z) / 2; 

        for(int j = 0; j < resolution[1]; j++) {
            for(int i = 0; i < resolution[0]; i++) {
                x = center.x + radius * cos(2 * PI * i / (resolution[0] - 1));
                y = min.y + j * y_gap;
                z = center.z + radius * sin(2 * PI * i / (resolution[0] - 1));

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
                y = (R + r * cos(phi)) * sin(theta) ;
                z = r * sin(phi);

                som_data(i, j, 0) = center + glm::vec3(x, y, z);
            }
        }
    } 
    else if(lattice_type == CUBE) {
        for(int j = 0; j < resolution[1]; j++) {
            for(int i = 0; i < resolution[0]; i++) {
                ratio_i = (double)i / (resolution[0] - 1);
                ratio_j = (double)j / (resolution[1] - 1);

                // y-z plane left
                x = min.x;
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
                y = min.y;
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
                z = min.z;
                som_data(i, j, 4) = glm::vec3(x, y, z);

                // x-y plane front
                x = min.x + size.x * ratio_i;
                y = min.y + size.y * ratio_j;
                z = max.z;
                som_data(i, j, 5) = glm::vec3(x, y, z);
            }
        }
    } 
    else cout<<"invalid lattices type number: "<<lattice_type<<endl;

    // initial texcoord
    static random_device rd;
    static std::mt19937 gen(rd());
    static uniform_real_distribution<double> dis(0.0, 1.0);

    for(int j = 0; j < resolution[1]; j++) {
        for(int i = 0; i < resolution[0]; i++) {
            som_data(i, j) = glm::vec3(dis(gen), dis(gen), 0.0f);
        }
    }
    for(int k = 0; k < 5; k++) { // 固定anchor points texcoord
        int i = anchor_points_idx[k].x;
        int j = anchor_points_idx[k].y;
        som_data(i, j) = glm::vec3(anchor_points_texcoord[k].x, anchor_points_texcoord[k].y, 0.0f);
    }
}

// 遍歷mesh找出BMU，距離最小的神經元index
glm::ivec3 SOM_c::findBMU(vector<glm::vec3> &mesh_data, glm::vec3 input_data)
{
    int n = 1;
    double minimum_distance = std::numeric_limits<double>::max();
    glm::ivec3 winner(0, 0, 0);
    
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_real_distribution<> dis(0.0, 1.0);

    for(int k = 0; k < resolution[2]; k++) {
        for(int j = 0; j < resolution[1]; j++) {
            for(int i = 0; i < resolution[0]; i++) {
                int idx = k * resolution[1] * resolution[0] + j * resolution[0] + i;
                glm::vec3 diff = mesh_data[idx] - input_data;
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

// 更新鄰域半徑
double SOM_c::updateNeighbourhoodRadius()
{
    return radius_init * exp((double)-iter_now * log(radius_init) / (double)iter_max);
}

// 更新學習率
double SOM_c::updateLearningRate()
{
    return learning_rate_init * exp((double)-iter_now / (double)iter_max);
}

// 專門算lattices = cube的index position
glm::ivec3 SOM_c::calculateCubeIndexPos(glm::ivec3 idx) {
    glm::ivec3 pos = {0, 0, 0};
    int face_idx = idx.z;

    if(face_idx == 0)
        pos = {0, idx.y, idx.x};
    else if(face_idx == 1)
        pos = {resolution[0] - 1, idx.y, idx.x};
    else if(face_idx == 2)
        pos = {idx.x, 0, idx.y};
    else if(face_idx == 3)
        pos = {idx.x, resolution[0] - 1, idx.y};
    else if(face_idx == 4)
        pos = {idx.x, idx.y, 0};
    else if(face_idx == 5)
        pos = {idx.x, idx.y, resolution[0] - 1};

    return pos;
}

// 計算BMU對其他神經元的影響力，距離越遠影響力越小，超過鄰域半徑的神經元不受BMU影響
double SOM_c::calculateInfluence(glm::ivec3 BMU, glm::ivec3 now_point)
{
    double distance = 0.0;
    if(lattice_type == SQUARE) {
        distance = length2(BMU - now_point);
    } 
    else if(lattice_type == CYLINDER) {
        double d1, d2;
        d1 = length2(BMU - now_point); // 平面上直線的算法

        if(BMU.x < now_point.x) // 跨越邊界的算法
            d2 = (resolution[0] + BMU.x - now_point.x) * (resolution[0] + BMU.x - now_point.x) + (BMU.y - now_point.y) * (BMU.y - now_point.y);
        else
            d2 = (resolution[0] + now_point.x - BMU.x) * (resolution[0] + now_point.x - BMU.x) + (BMU.y - now_point.y) * (BMU.y - now_point.y);
        
        distance = (d1 < d2) ? d1 : d2;
    }
    else if(lattice_type == DONUT) {
        double d1, d2, d3, d4;
        d1 = length2(BMU - now_point); // 平面上直線的算法

        if(BMU.x < now_point.x) // 跨越x邊界的算法
            d2 = (resolution[0] + BMU.x - now_point.x) * (resolution[0] + BMU.x - now_point.x) + (BMU.y - now_point.y) * (BMU.y - now_point.y);
        else
            d2 = (resolution[0] + now_point.x - BMU.x) * (resolution[0] + now_point.x - BMU.x) + (BMU.y - now_point.y) * (BMU.y - now_point.y);

        if(BMU.y < now_point.y) // 跨越y邊界的算法
            d3 = (resolution[1] + BMU.y - now_point.y) * (resolution[1] + BMU.y - now_point.y) + (BMU.x - now_point.x) * (BMU.x - now_point.x);
        else
            d3 = (resolution[1] + now_point.y - BMU.y) * (resolution[1] + now_point.y - BMU.y) + (BMU.x - now_point.x) * (BMU.x - now_point.x);

        if(BMU.x < now_point.x && BMU.y < now_point.y) // 跨越x和y邊界的算法
            d4 = (resolution[0] + BMU.x - now_point.x) * (resolution[0] + BMU.x - now_point.x) + (resolution[1] + BMU.y - now_point.y) * (resolution[1] + BMU.y - now_point.y);
        else if(BMU.x < now_point.x)
            d4 = (resolution[0] + BMU.x - now_point.x) * (resolution[0] + BMU.x - now_point.x) + (resolution[1] + now_point.y - BMU.y) * (resolution[1] + now_point.y - BMU.y);
        else if(BMU.y < now_point.y)
            d4 = (resolution[0] + now_point.x - BMU.x) * (resolution[0] + now_point.x - BMU.x) + (resolution[1] + BMU.y - now_point.y) * (resolution[1] + BMU.y - now_point.y);
        else 
            d4 = (resolution[0] + now_point.x - BMU.x) * (resolution[0] + now_point.x - BMU.x) + (resolution[1] + now_point.y - BMU.y) * (resolution[1] + now_point.y - BMU.y);
    
        distance = min(min(d1, d2), min(d3, d4));  
    }
    else if(lattice_type == CUBE) {
        distance = length2(calculateCubeIndexPos(BMU) - calculateCubeIndexPos(now_point));
    }

    if(distance > radius_now * radius_now)
        return 0.0;
    return exp(-(distance / (2 * radius_now * radius_now)));
}

// void CreateThread()
// {
//     if (t1.joinable())
//     {
//         t1.join();
//     }
//     t1 = std::thread(RunSOM);
//     // t1.detach(); // if you want to run it in background
// }

// 丟進mesh資料和訓練輸入來進行一次訓練
void SOM_c::runSOM(vector<glm::vec3> &mesh_data, glm::vec3 input_data)
{
    // 1. pick a random input vector

    // 2. find the best matching unit (BMU)
    glm::ivec3 BMU = findBMU(mesh_data, input_data);
    
    // 3. update the radius and learning rate
    radius_now = updateNeighbourhoodRadius();
    learning_rate_now = updateLearningRate();
    
    // 4. update the neighbor nodes
    #pragma omp parallel for collapse(3)
    for(int k = 0; k < resolution[2]; k++) {
        for(int j = 0; j < resolution[1]; j++) {
            for(int i = 0; i < resolution[0]; i++) {
                int idx = k * resolution[1] * resolution[0] + j * resolution[0] + i;
                glm::ivec3 now_point(i, j, k);
                double influence = calculateInfluence(BMU, now_point);
                if(influence < 1e-5)
                    influence = 0;
                mesh_data[idx] += (float)influence * (float)learning_rate_now * (input_data - mesh_data[idx]);
            }
        }
    }
    iter_now++;
}