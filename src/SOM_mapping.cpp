#include <chrono>
#include "SOM_mapping.h"

using namespace chrono;

TexFilter tex_filter = LINEAR;
SizeFilter size_filter = MAGNIFICATION;
glm::ivec3 offset[5] = {{1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}, {1, 0, 0}}; // for BMU相鄰的三角形

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

// porjp是不是在o(BMU), a, b連成的三角形中
bool inTriangle(glm::vec3 a, glm::vec3 b, glm::vec3 o, glm::vec3 projp) {
    // 法1: 用面積比
    float area_ab = glm::length(glm::cross(a - projp, b - projp));
    float area_bo = glm::length(glm::cross(b - projp, o - projp));
    float area_oa = glm::length(glm::cross(o - projp, a - projp));
    float total_area = glm::length(glm::cross(a - o, b - o));

    if(fabs(total_area - (area_ab + area_bo + area_oa)) < 1e-5f) // 面積加起來大約相等
        return true;
    return false;
    // 法2: 在內部代表被projp分三塊的三角形法向量都要同向
    // glm::vec3 normal_ab = glm::cross(a - projp, b - projp);
    // glm::vec3 normal_bo = glm::cross(b - projp, o - projp);
    // glm::vec3 normal_oa = glm::cross(o - projp, a - projp);
    // glm::vec3 normal = glm::cross(a - o, b - o);
    // float c1 = glm::dot(normal_ab, normal);
    // float c2 = glm::dot(normal_bo, normal);
    // float c3 = glm::dot(normal_oa, normal);
    // if ((c1 >= 0.0f && c2 >= 0.0f && c3 >= 0.0f) || 
    //     (c1 <= 0.0f && c2 <= 0.0f && c3 <= 0.0f)) // 處理了面積為0的情況
    //     return true;
    // return false;
}

// 投影點在三角形外面就投影到邊上
glm::vec3 outerProjp(glm::vec3 a, glm::vec3 b, glm::vec3 o, glm::vec3 projp) {
    glm::vec3 projp_on_a, projp_on_b;
    double dist_a, dist_b;
    glm::vec3 oprojp = projp - o;

    // assume在oa外
    glm::vec3 oa = a - o;
    double length2 = oa.x * oa.x + oa.y * oa.y;
    float r = glm::dot(oa, oprojp) / length2;
    if(r > 1.0f)
        projp_on_a = a;
    else if(r < 0.0f)
        projp_on_a = o;
    else
        projp_on_a = o + r * oa;
    dist_a = glm::length(projp_on_a - projp); // 跟原本的projp比會比跟BMU比準，因為projp垂直三角形，BMU沒有，無法推斷正確距離
    
    // assume在ob外
    glm::vec3 ob = b - o;
    length2 = ob.x * ob.x + ob.y * ob.y;
    r = glm::dot(ob, oprojp) / length2;
    if(r > 1.0f)
        projp_on_b = b;
    else if(r < 0.0f)
        projp_on_b = o;
    else
        projp_on_b = o + r * ob;
    dist_b = glm::length(projp_on_b - projp);

    if(dist_a < dist_b)
        return projp_on_a;
    else 
        return projp_on_b;
}

// 找正確的投影點
glm::vec3 getTriangleProjectionPoint(glm::vec3 a, glm::vec3 b, glm::vec3 o, glm::vec3 p, SomData &som_data) {
    glm::vec3 normal = glm::normalize(glm::cross(a - o, b - o));
    glm::vec3 op = p - o;

    // find initial projection point
    glm::vec3 d = -(glm::dot(op, normal)) * normal;
    glm::vec3 projp = p + d;

    // check projection point
    if(!inTriangle(a, b, o, projp))
        projp = outerProjp(a, b, o, projp);
    return projp;
}

// 用來判斷和BMU一起構成三角形的其他兩點是否在邊界內，在邊界內就順便改成對應的索引
bool isNodeIndexValid(SomData som_data, glm::ivec3 &A) {
    if(A.x < 0 || A.x > som_data.resolution[0] - 1 || A.y < 0 || A.y > som_data.resolution[1] - 1) { // x or y axis out of bound
        if(som_data.lattice_type == SQUARE) {
            return false;
        } 
        else if(som_data.lattice_type == CYLINDER) { 
            if(A.y < 0 || A.y > som_data.resolution[1] - 1) // y axis out of bound
                return false;
            if(A.x < 0) // 其實只會是 A.x == -1
                A.x += som_data.resolution[0] - 1; // 要位移 width - 1 才會到對應的node index
            else if(A.x > som_data.resolution[0] - 1)
                A.x -= som_data.resolution[0] - 1;
        }
        else if(som_data.lattice_type == DONUT) {
            if(A.x < 0)
                A.x += som_data.resolution[0] - 1;
            else if(A.x > som_data.resolution[0] - 1)
                A.x -= som_data.resolution[0] - 1;
            else if(A.y < 0) // 因為鄰居只會偏移x or y其中一個，所以不需要設成if
                A.y += som_data.resolution[1] - 1;
            else if(A.y > som_data.resolution[1] - 1)
                A.y -= som_data.resolution[1] - 1;
        } 
        else if(som_data.lattice_type == CUBE) { 
            int w = som_data.resolution[0];
            int h = som_data.resolution[1];
            glm::ivec3 old_A = A;
            if(old_A.x < 0) {
                static int face[] = {4, 4, 0, 0, 0, 0};
                A.z = face[old_A.z];
                if(old_A.z == 0 || old_A.z == 4)
                    A.x = 1;
                else if(old_A.z == 1 || old_A.z == 5)
                    A.x = w - 2;
                else if(old_A.z == 2) {
                    A.x = old_A.y;
                    A.y = 1;
                } else if(old_A.z == 3) {
                    A.x = old_A.y;
                    A.y = h - 2;
                }
            }
            else if (old_A.x > w - 1) {
                static int face[] = {5, 5, 1, 1, 1, 1};
                A.z = face[old_A.z];

                if (old_A.z == 0 || old_A.z == 4) {
                    A.x = 1;
                    A.y = old_A.y;
                } else if (old_A.z == 1 || old_A.z == 5) {
                    A.x = w - 2;
                    A.y = old_A.y;
                } else if (old_A.z == 2) {
                    A.x = old_A.y;
                    A.y = 1;
                } else if (old_A.z == 3) {
                    A.x = old_A.y;
                    A.y = h - 2;
                }
            }
            else if (old_A.y < 0) {
                static int face[] = {2, 2, 4, 4, 2, 2};
                A.z = face[old_A.z];

                if (old_A.z == 0) {
                    A.x = 1;
                    A.y = old_A.x;
                } else if (old_A.z == 1) {
                    A.x = w - 2;
                    A.y = old_A.x;
                } else if (old_A.z == 2 || old_A.z == 4) {
                    A.x = old_A.x;
                    A.y = 1;
                } else if (old_A.z == 3 || old_A.z == 5) {
                    A.x = old_A.x;
                    A.y = h - 2;
                }
            }
            else if (old_A.y > h - 1) {
                static int face[] = {3, 3, 5, 5, 3, 3};
                A.z = face[old_A.z];

                if (old_A.z == 0) {
                    A.x = 1;
                    A.y = old_A.x;
                } else if (old_A.z == 1) {
                    A.x = w - 2;
                    A.y = old_A.x;
                } else if (old_A.z == 2 || old_A.z == 4) {
                    A.x = old_A.x;
                    A.y = 1;
                } else if (old_A.z == 3 || old_A.z == 5) {
                    A.x = old_A.x;
                    A.y = h - 2;
                }
            }
            // face->face(i, j)
            // if x < 0
            // 0->4(1, y) 1->4(w-2, y)
            // 2->0(y, 1) 3->0(y, h-2)
            // 4->0(1, y) 5->0(w-2, y)
            // else if x > w - 1
            // 0->5(1, y) 1->5(w-2, y)
            // 2->1(y, 1) 3->1(y, h-2)
            // 4->1(1, y) 5->1(w-2, y)
            // if y < 0
            // 0->2(1, x) 1->2(w-2, x)
            // 2->4(x, 1) 3->4(x, h-2)
            // 4->2(x, 1) 5->2(x, h-2)
            // else if y > h - 1
            // 0->3(1, x) 1->3(w-2, x)
            // 2->5(x, 1) 3->5(x, h-2)
            // 4->3(x, 1) 5->3(x, h-2)
        }
    }
    return true;
}

// 在BMU鄰近三角片中找與input最短距離的投影點的texcoord
glm::vec2 SOMMapping(glm::vec3 voxel, SomData &som_data) {
    glm::ivec3 BMU = findPosBMU(som_data, voxel);
    double min_distance = numeric_limits<double>::max();
    glm::vec2 texcoord;
    for(int i = 0; i < 4; i++) {
        // 三角形頂點index
        glm::ivec3 A = BMU + offset[i];
        glm::ivec3 B = BMU + offset[i + 1];
        // 超出範圍就跳過這個三角形
        if(!isNodeIndexValid(som_data, A) || !isNodeIndexValid(som_data, B))
            continue;
        // 取得三角形位置
        glm::vec3 a = som_data(A.x, A.y, A.z);
        glm::vec3 b = som_data(B.x, B.y, B.z);
        glm::vec3 o = som_data(BMU.x, BMU.y, BMU.z);
        // 得到投影點
        glm::vec3 projp = getTriangleProjectionPoint(a, b, o, voxel, som_data);

        double distance = (projp.x - voxel.x) * (projp.x - voxel.x) + (projp.y - voxel.y) * (projp.y - voxel.y) + (projp.z - voxel.z) * (projp.z - voxel.z);
        if(distance >= min_distance) // 比現在最小距離值大代表不採用這個三角形的投影點
            continue;
        min_distance = distance;

        // 按照三角形頂點與面積計算texcoord
        float area_ab = glm::length(glm::cross(a - projp, b - projp));
        float area_bo = glm::length(glm::cross(b - projp, o - projp));
        float area_oa = glm::length(glm::cross(o - projp, a - projp));
        float total_area = glm::length(glm::cross(a - o, b - o));
        texcoord = som_data(BMU.x, BMU.y) * (area_ab / total_area) + som_data(A.x, A.y) * (area_bo / total_area) + som_data(B.x, B.y) * (area_oa / total_area);
    }
    return texcoord;
}

// 對每一層的體素都做兩階段映射算出顏色值
void twoStepMapping(LayerVoxels &layer_voxels, SomData &som_data, vector<vector<glm::vec3>> &image) {
    auto start = steady_clock::now();
    int size = layer_voxels.pos.size();
    layer_voxels.color.resize(size);
    int height = image.size();
    int width = image[0].size();

    for(int i = 0; i < size; i++) {
        // SOM mapping
        glm::vec3 voxel_pos = layer_voxels.pos[i];
        glm::vec2 texcoord = SOMMapping(voxel_pos + glm::vec3{0.5f, 0.5f, 0.5f}, som_data); // 體素中心點映射後的texcoord
        // texture mapping
        if(tex_filter == NEAREST) { 
            glm::ivec2 texel_index = (texcoord * glm::vec2(width - 1, height - 1) + 0.5f); // 四捨五入 + 轉型
            // 取得0~255值後四捨五入，再轉回0~1
            layer_voxels.color[i] = image[texel_index.y][texel_index.x];
        } 
        else if(tex_filter == LINEAR) { 
            if(width * height < size)
                size_filter = MAGNIFICATION;
            else 
                size_filter = MINIFICATION;
            glm::vec2 texel_index = (texcoord * glm::vec2(width - 1, height - 1));
            if(size_filter == MAGNIFICATION) {
                glm::ivec2 texel_IJ = (glm::ivec2)texel_index; // 左下角texel
                float dx = texel_index.x - texel_IJ.x;
                float dy = texel_index.y - texel_IJ.y;
                int offset[2] = {1, 1}; // x和y方向index位移
                if(texel_IJ.x == width - 1)
                    offset[0] = 0;
                if(texel_IJ.y == height - 1)
                    offset[1] = 0;
                glm::vec3 c00 = image[texel_IJ.y][texel_IJ.x];
                glm::vec3 c10 = image[texel_IJ.y][texel_IJ.x + offset[0]];
                glm::vec3 c11 = image[texel_IJ.y + offset[1]][texel_IJ.x + offset[1]];
                glm::vec3 c01 = image[texel_IJ.y + offset[1]][texel_IJ.x];
                layer_voxels.color[i] = (1 - dx) * (1 - dy) * c00 + dx * (1 - dy) * c10 + (1 - dx) * dy * c01 + dx * dy * c11;
            }
            else if(size_filter == MINIFICATION) {
                // texcoord
                glm::vec2 temp0 = SOMMapping(voxel_pos, som_data); // 體素左下後點映射後的texcoord
                glm::vec2 temp1 = SOMMapping(voxel_pos + glm::vec3{1.0f, 1.0f, 1.0f}, som_data); // 體素有上前點映射後的texcoord
                glm::vec2 texcoord_000 = glm::min(temp0, temp1);
                glm::vec2 texcoord_111 = glm::max(temp0, temp1);
                // texture map index // 四捨五入 + 轉型
                glm::ivec2 texel_index_000 = texcoord_000 * glm::vec2(width - 1, height - 1);
                glm::ivec2 texel_index_111 = (texcoord_111 * glm::vec2(width - 1, height - 1)) + 0.5f;
                // start and end point of texture map index
                glm::ivec2 t0 = glm::min(texel_index_000, texel_index_111);
                glm::ivec2 t1 = glm::max(texel_index_000, texel_index_111);
                // box filter
                int num = 0;
                glm::vec3 color = {0.0f, 0.0f, 0.0f};
                for(int j = t0.y; j <= t1.y; j++) {
                    for(int i = t0.x; i <= t1.x; i++) {
                        color += image[j][i];
                        num++;
                    }
                }
                color /= num;
                layer_voxels.color[i] = color;
                // 另一種想法:
                // 理想情況，SOM最後完美貼合模型，模型也平均散開成表面沒有體素被包住，張開來變成兩張長寬比相同但不同解析度的平面在映射 
                // -> (width * ratio) * (height * ratio) = #(voxels) -> ratio = sqrt(#(voxels) / (width * height))
            }
        }
    }
    cout<<"texture filter: ";
    if(size_filter == MAGNIFICATION)
        cout<<"MAGNIFICATION"<<endl;
    else if(size_filter == MINIFICATION)
        cout<<"MINIFICATION"<<endl;
    auto end = steady_clock::now();
    auto diff = duration<double>(end - start);
    cout<<"兩階段映射時間: "<<diff.count()<<" s\n"<<endl;
}