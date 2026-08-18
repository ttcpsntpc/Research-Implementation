#include <cfloat>
#include <cmath>
#include <chrono>
#include <queue>
#include "distance_field_3D.h"

using namespace chrono;

bool DistanceField3D::isRangeValid(int i, int j, int k) {
    return (i >= 0 && i < width && j >= 0 && j < height && k >= 0 && k < depth);
}

DistanceFieldData DistanceField3D::compute_LL(const vector<tuple<int, int, int>>& points, int width, int height, int depth) {
    DistanceFieldData df_data;
    Close_LL *close_list_header = nullptr;
    // 距離值沒必要初始化，主要用state判斷
    // initialize voxel state
    max_distance = 0;
    df_data.width = width;
    df_data.height = height;
    df_data.depth = depth;
    this->width = width;
    this->height = height;
    this->depth = depth;

    df_data.voxel_distance.resize(width * height * depth);
    voxel_state.assign(width * height * depth, INITIAL);
    
    // initialize model's voxel
    for(auto& p : points) {
        int x = get<0>(p);
        int y = get<1>(p);
        int z = get<2>(p);
        df_data(x, y, z) = 0;
        voxel_state[idx(x, y, z)] = DONE;
    }
    // initialize other voxel
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; j++) {
            for(int k = 0; k < depth; k++) {
                // 看是不是done
                if(voxel_state[idx(i, j, k)] == DONE)
                    continue; 
                // 看是不是close
                else if(isClose(i, j, k)) {
                    voxel_state[idx(i, j, k)] = CLOSE;
                    // 計算距離
                    df_data(i, j, k) = computeDistance(df_data, i, j, k);
                    // 加入close list
                    insertList(i, j, k, df_data(i, j, k), close_list_header);
                }
                else {
                    voxel_state[idx(i, j, k)] = FAR;
                }
            }
        }
    }
    // Fast Marching Method 
    while(close_list_header != nullptr) {
        // choose closest in clos list, and make it DONE
        Close_LL *closest = close_list_header;
        close_list_header = close_list_header->next;
        int i = closest->i;
        int j = closest->j;
        int k = closest->k;
        voxel_state[idx(i, j, k)] = DONE;
        max_distance = (closest->distance > max_distance) ? closest->distance : max_distance;
        delete closest;
        // update neighbors
        for(int dir = 0; dir < 6; dir++) {
            int x = i + offsets[dir][0];
            int y = j + offsets[dir][1];
            int z = k + offsets[dir][2];
            if(!isRangeValid(x, y, z))
                continue;
            if(voxel_state[idx(x, y, z)] == DONE)
                continue;
            else if(voxel_state[idx(x, y, z)] == CLOSE) {
                // recompute distance
                double new_distance = computeDistance(df_data, x, y, z);
                if(new_distance < df_data(x, y, z)) {
                    df_data(x, y, z) = new_distance;
                    // remove old node
                    Close_LL *current = close_list_header;
                    Close_LL *previous = nullptr;
                    while(current->i != x || current->j != y || current->k != z) {
                        previous = current;
                        current = current->next;
                    }
                    if(current == close_list_header) {
                        close_list_header = close_list_header->next;
                    }
                    else {
                        previous->next = current->next;
                    }
                    delete current;
                    // insert new node
                    insertList(x, y, z, new_distance, close_list_header);
                }
            }
            else if(voxel_state[idx(x, y, z)] == FAR) {
                voxel_state[idx(x, y, z)] = CLOSE;
                double distance = computeDistance(df_data, x, y, z);
                df_data(x, y, z) = distance;
                insertList(x, y, z, distance, close_list_header);
            }
        }
    }
    df_data.max_distance = max_distance;

    // 最後判斷模型內外
    determineInOut(df_data.voxel_distance);

    return df_data;
}

DistanceFieldData DistanceField3D::compute_LL(const vector<unsigned char>& voxels, int width, int height, int depth) {
    DistanceFieldData df_data;
    Close_LL *close_list_header = nullptr;
    // 距離值沒必要初始化，主要用state判斷
    // initialize voxel state
    max_distance = 0;
    df_data.width = width;
    df_data.height = height;
    df_data.depth = depth;
    this->width = width;
    this->height = height;
    this->depth = depth;

    df_data.voxel_distance.resize(width * height * depth);
    voxel_state.assign(width * height * depth, INITIAL);
    
    auto start = steady_clock::now();

    // initialize model's voxel
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; j++) {
            for(int k = 0; k < depth; k++) {
                if(voxels[idx(i, j, k)] == 255) {
                    df_data(i, j, k) = 0;
                    voxel_state[idx(i, j, k)] = DONE;
                }
            }
        }
    }
    
    // initialize other voxel
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; j++) {
            for(int k = 0; k < depth; k++) {
                // 看是不是done
                if(voxel_state[idx(i, j, k)] == DONE)
                    continue; 
                // 看應不應該放入close
                else if(isClose(i, j, k)) {
                    voxel_state[idx(i, j, k)] = CLOSE;
                    // 計算距離
                    df_data(i, j, k) = computeDistance(df_data, i, j, k);
                    // 加入close list
                    insertList(i, j, k, df_data(i, j, k), close_list_header);
                }
                else {
                    voxel_state[idx(i, j, k)] = FAR;
                }
            }
        }
    }
    
    // Fast Marching Method 
    while(close_list_header != nullptr) {
        // choose closest in close list, and make it DONE
        Close_LL *closest = close_list_header;
        close_list_header = close_list_header->next;
        int i = closest->i;
        int j = closest->j;
        int k = closest->k;
        voxel_state[idx(i, j, k)] = DONE;
        max_distance = (closest->distance > max_distance) ? closest->distance : max_distance;
        delete closest;
        // update neighbors
        for(int dir = 0; dir < 6; dir++) {
            int x = i + offsets[dir][0];
            int y = j + offsets[dir][1];
            int z = k + offsets[dir][2];
            if(!isRangeValid(x, y, z))
                continue;
            if(voxel_state[idx(x, y, z)] == DONE)
                continue;
            else if(voxel_state[idx(x, y, z)] == CLOSE) {
                // recompute distance
                double new_distance = computeDistance(df_data, x, y, z);
                if(new_distance < df_data(i, j, k)) {
                    // recompute distance
                    double new_distance = computeDistance(df_data, x, y, z);
                    if(new_distance < df_data(i, j, k)) {
                        df_data(i, j, k) = new_distance;
                        // 不用移除在Close中的舊值，因為新的距離小先算完後就變成DONE而自然的跳過
                        // insert new node
                        insertList(x, y, z, new_distance, close_list_header);
                    }
                }
            }
            else if(voxel_state[idx(x, y, z)] == FAR) {
                voxel_state[idx(x, y, z)] = CLOSE;
                double distance = computeDistance(df_data, x, y, z);
                df_data(i, j, k) = distance;
                insertList(x, y, z, distance, close_list_header);
            }
        }
    }
    df_data.max_distance = max_distance;

    // 最後判斷模型內外
    determineInOut(df_data.voxel_distance);

    auto end = steady_clock::now();
    auto diff = duration_cast<milliseconds>(end - start);
    cout<<"距離場計算時間: "<<diff.count() / 1000<<"s"<<endl;
    cout<<"最大距離值"<<max_distance<<endl;

    return df_data;
}

void DistanceField3D::insertList(int i, int j, int k, double distance, Close_LL *&close_list_header) {
    Close_LL *new_node = new Close_LL{i, j, k, distance, nullptr};
    if(close_list_header == nullptr) {
        close_list_header = new_node;
        return;
    }
    Close_LL *current = close_list_header;
    Close_LL *previous = nullptr;
    while(current != nullptr && current->distance <= distance) {
        previous = current;
        current = current->next;
    }
    if(previous == nullptr) { // insert at head
        new_node->next = close_list_header;
        close_list_header = new_node;
    }
    else {
        previous->next = new_node;
        new_node->next = current;
    }
}

DistanceFieldData DistanceField3D::compute_PQ(const vector<unsigned char>& voxels, int width, int height, int depth) {
    DistanceFieldData df_data;
    // 距離值沒必要初始化，主要用state判斷
    // initialize voxel state
    max_distance = 0;
    df_data.width = width;
    df_data.height = height;
    df_data.depth = depth;
    this->width = width;
    this->height = height;
    this->depth = depth;

    df_data.voxel_distance.resize(width * height * depth);
    voxel_state.assign(width * height * depth, INITIAL);
    
    priority_queue<Close_PQ, vector<Close_PQ>, greater<Close_PQ>> Close_heap;

    auto start = steady_clock::now();

    // initialize model's voxel
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; j++) {
            for(int k = 0; k < depth; k++) {
                if(voxels[idx(i, j, k)] == 255) {
                    df_data(i, j, k) = 0;
                    voxel_state[idx(i, j, k)] = DONE;
                }
            }
        }
    }
    
    // initialize other voxel
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; j++) {
            for(int k = 0; k < depth; k++) {
                // 看是不是done
                if(voxel_state[idx(i, j, k)] == DONE)
                    continue; 
                // 看應不應該放入close
                else if(isClose(i, j, k)) {
                    voxel_state[idx(i, j, k)] = CLOSE;
                    // 計算距離
                    df_data(i, j, k) = computeDistance(df_data, i, j, k);
                    Close_heap.push({i, j, k, df_data(i, j, k)});
                }
                else {
                    voxel_state[idx(i, j, k)] = FAR;
                }
            }
        }
    }
    
    // Fast Marching Method 
    while(!Close_heap.empty()) {
        // choose closest in close heap, and make it DONE
        Close_PQ closest = Close_heap.top();
        Close_heap.pop();

        int i = closest.i;
        int j = closest.j;
        int k = closest.k;
        // 如果這個體素已經被處理過了（可能被更短的路徑捷足先登），則直接跳過
        if(voxel_state[idx(i, j, k)] == DONE) {
            continue;
        }

        voxel_state[idx(i, j, k)] = DONE;
        max_distance = (closest.distance > max_distance) ? closest.distance : max_distance;
        
        // update neighbors
        for(int dir = 0; dir < 6; dir++) {
            int x = i + offsets[dir][0];
            int y = j + offsets[dir][1];
            int z = k + offsets[dir][2];
            if(!isRangeValid(x, y, z))
                continue;
            if(voxel_state[idx(x, y, z)] == DONE)
                continue;
            else if(voxel_state[idx(x, y, z)] == CLOSE) {
                // recompute distance
                double new_distance = computeDistance(df_data, x, y, z);
                if(new_distance < df_data(x, y, z)) {
                    df_data(x, y, z) = new_distance;
                    // 不用移除舊節點
                    // 直接將同個體素但更短的距離push進去
                    Close_heap.push({x, y, z, new_distance});
                }
            }
            else if(voxel_state[idx(x, y, z)] == FAR) {
                voxel_state[idx(x, y, z)] = CLOSE;
                double distance = computeDistance(df_data, x, y, z);
                df_data(x, y, z) = distance;
                Close_heap.push({x, y, z, distance});
            }
        }
    }
    df_data.max_distance = max_distance;

    // 最後判斷模型內外
    determineInOut(df_data.voxel_distance);
    
    auto end = steady_clock::now();
    auto diff = duration_cast<milliseconds>(end - start);
    cout<<"距離場計算時間: "<<diff.count() / 1000<<"s"<<endl;
    cout<<"最大距離值"<<max_distance<<endl;

    return df_data;
}

double DistanceField3D::computeDistance(DistanceFieldData &df_data, int i, int j, int k) {
    double At[3], Bt[3], Ct[3], at[3], bt[3], ct[3], a, b, c, distance;
    for(int dir = 0; dir < 3; dir++) { // 0:x方向, 1:y方向, 2:z方向
        // 定義某個方向的左右兩個鄰居的索引v0和v2 (自己是v1)
        int v0[3], v2[3]; 
        v0[0] = i + offsets[dir * 2][0];
        v0[1] = j + offsets[dir * 2][1];
        v0[2] = k + offsets[dir * 2][2];
        v2[0] = i + offsets[dir * 2 + 1][0];
        v2[1] = j + offsets[dir * 2 + 1][1];
        v2[2] = k + offsets[dir * 2 + 1][2];

        // double[-1][0]可能沒事，但vector[-1][0]會出錯，所以取值要先檢查
        double u0 = isRangeValid(v0[0], v0[1], v0[2]) ? df_data(v0[0], v0[1], v0[2]) : DBL_MAX; 
        double u2 = isRangeValid(v2[0], v2[1], v2[2]) ? df_data(v2[0], v2[1], v2[2]) : DBL_MAX; 
        if ((v0[0] < 0 || v0[1] < 0 || v0[2] < 0) && (v2[0] >= width || v2[1] >= height || v2[2] >= depth)) { // 兩邊都超出範圍(就是2D的情況)
            At[dir] = 0;
            Bt[dir] = 0;
            Ct[dir] = 0;
        }
        else if(v0[0] < 0 || v0[1] < 0 || v0[2] < 0) { // x或y或z小於範圍
            if(voxel_state[idx(v2[0], v2[1], v2[2])] != DONE) {
                At[dir] = 0;
                Bt[dir] = 0;
                Ct[dir] = 0;
            }
            else {
                At[dir] = 0;
                Bt[dir] = -1;
                Ct[dir] = 1;
            }
        }
        else if(v2[0] >= width || v2[1] >= height || v2[2] >= depth) { // x或y或z大於範圍
            if(voxel_state[idx(v0[0], v0[1], v0[2])] != DONE) {
                At[dir] = 0;
                Bt[dir] = 0;
                Ct[dir] = 0;
            }
            else {
                At[dir] = -1;
                Bt[dir] = 1;
                Ct[dir] = 0;
            }
        } 
        else { // 正常判斷
            if(voxel_state[idx(v0[0], v0[1], v0[2])] != DONE && voxel_state[idx(v2[0], v2[1], v2[2])] != DONE) {
                At[dir] = 0;
                Bt[dir] = 0;
                Ct[dir] = 0;
            }
            else if(voxel_state[idx(v0[0], v0[1], v0[2])] == DONE && voxel_state[idx(v2[0], v2[1], v2[2])] != DONE) {
                At[dir] = -1;
                Bt[dir] = 1;
                Ct[dir] = 0;
            }
            else if(voxel_state[idx(v0[0], v0[1], v0[2])] != DONE && voxel_state[idx(v2[0], v2[1], v2[2])] == DONE) {
                At[dir] = 0;
                Bt[dir] = -1;
                Ct[dir] = 1;
            }
            else { // both DONE
                if(u0 <= u2) {
                    At[dir] = -1;
                    Bt[dir] = 1;
                    Ct[dir] = 0;
                }
                else {
                    At[dir] = 0;
                    Bt[dir] = -1;
                    Ct[dir] = 1;
                }
            }
        }
        at[dir] = Bt[dir]*Bt[dir];
        bt[dir] = 2*(At[dir]*Bt[dir]*u0 + Bt[dir]*Ct[dir]*u2);
        ct[dir] = At[dir]*At[dir]*u0*u0 + Ct[dir]*Ct[dir]*u2*u2;
    }
    a = at[0] + at[1] + at[2];
    b = bt[0] + bt[1] + bt[2];
    c = ct[0] + ct[1] + ct[2] - 1;
    distance = (-b + sqrt(b*b - 4*a*c)) / (2*a);
    return distance;
}

bool DistanceField3D::isClose(int i, int j, int k) {
    for (int dir = 0; dir < 6; dir++) {
        int x = i + offsets[dir][0];
        int y = j + offsets[dir][1];
        int z = k + offsets[dir][2];
        if (isRangeValid(x, y, z) && voxel_state[idx(x, y, z)] == DONE) {
            return true;
        }
    }
    return false;
}

// 分辨內外距離，比如分別從x=0和x=width開始，往內走，並且把距離值乘上-1，如果遇到done就切換in/out狀態
void DistanceField3D::determineInOut(vector<double> &voxel_distance) {
    vector<int> voxel_out_count = vector<int>(width * height * depth, 0); // 紀錄每個voxel被判斷為模型外部的次數
    vector<double> voxel_temp1 = voxel_distance, voxel_temp2 = voxel_distance;
    // 做x方向的判斷
    for(int j = 0; j < height; j++) {
        for(int k = 0; k < depth; k++) {
            // 從x=0開始往內走
            bool is_out = true;
            for(int i = 0; i < width; i++) {
                  // 遇到模型就切換
                if(voxel_distance[idx(i, j, k)] == 0 && is_out == true) { // 第一次碰到模型才切換
                    is_out = !is_out;
                    continue;
                } else if(voxel_distance[idx(i, j, k)] == 0) continue; // 第二次以後碰到模型直接跳過

                if(!is_out) { // 到內部時先判斷鄰居是不是外部，只要有一個是外部就視為外部
                    int dir;
                    for(dir = 0; dir < 6; dir++) {
                        int x = i + offsets[dir][0];
                        int y = j + offsets[dir][1];
                        int z = k + offsets[dir][2];
                        if(isRangeValid(x, y, z) && voxel_temp1[idx(x, y, z)] < 0) {
                            break;
                        }
                    }
                    if(dir == 6) 
                        continue; // 如果鄰居都不是外部才確定自己是內部，保持正值，繼續往內走
                }
                // 在外部就count++並將距離值變負的
                voxel_out_count[idx(i, j, k)]++;
                voxel_temp1[idx(i, j, k)] = -voxel_temp1[idx(i, j, k)];
            }
        }
        for(int k = depth - 1; k >= 0; k--) {
            // 從x=width - 1開始往內走
            bool is_out = true;
            for(int i = width - 1; i >= 0; i--) {
                  // 遇到模型就切換
                if(voxel_distance[idx(i, j, k)] == 0 && is_out == true) { // 第一次碰到模型才切換
                    is_out = !is_out;
                    continue;
                } else if(voxel_distance[idx(i, j, k)] == 0) continue; // 第二次以後碰到模型直接跳過

                if(!is_out) { // 到內部時先判斷鄰居是不是外部，只要有一個是外部就視為外部
                    int dir;
                    for(dir = 0; dir < 6; dir++) {
                        int x = i + offsets[dir][0];
                        int y = j + offsets[dir][1];
                        int z = k + offsets[dir][2];
                        if(isRangeValid(x, y, z) && voxel_temp2[idx(x, y, z)] < 0) {
                            break;
                        }
                    }
                    if(dir == 6) 
                        continue; // 如果鄰居都不是外部才確定自己是內部，保持正值，繼續往內走
                }
                // 在外部就count++並距離值變負的
                voxel_out_count[idx(i, j, k)]++;
                voxel_temp2[idx(i, j, k)] = -voxel_temp2[idx(i, j, k)];
            }
        }
    }

    voxel_temp1 = voxel_distance, voxel_temp2 = voxel_distance;
    // 做y方向的判斷
    for(int i = 0; i < width; i++) {
        for(int k = 0; k < depth; k++) {
            // 從y=0開始往內走
            bool is_out = true;
            for(int j = 0; j < height; j++) {
                  // 遇到模型就切換
                if(voxel_distance[idx(i, j, k)] == 0 && is_out == true) { // 第一次碰到模型才切換
                    is_out = !is_out;
                    continue;
                } else if(voxel_distance[idx(i, j, k)] == 0) continue; // 第二次以後碰到模型直接跳過

                if(!is_out) { // 到內部時先判斷鄰居是不是外部，只要有一個是外部就視為外部
                    int dir;
                    for(dir = 0; dir < 6; dir++) {
                        int x = i + offsets[dir][0];
                        int y = j + offsets[dir][1];
                        int z = k + offsets[dir][2];
                        if(isRangeValid(x, y, z) && voxel_temp1[idx(x, y, z)] < 0) {
                            break;
                        }
                    }
                    if(dir == 6) 
                        continue; // 如果鄰居都不是外部才確定自己是內部，保持正值，繼續往內走
                }
                // 在外部就count++並距離值變負的
                voxel_out_count[idx(i, j, k)]++;
                voxel_temp1[idx(i, j, k)] = -voxel_temp1[idx(i, j, k)];
            }
        }
        for(int k = depth - 1; k >= 0; k--) {
            // 從y=height - 1開始往內走
            bool is_out = true;
            for(int j = height - 1; j >= 0; j--) {
                  // 遇到模型就切換
                if(voxel_distance[idx(i, j, k)] == 0 && is_out == true) { // 第一次碰到模型才切換
                    is_out = !is_out;
                    continue;
                } else if(voxel_distance[idx(i, j, k)] == 0) continue; // 第二次以後碰到模型直接跳過

                if(!is_out) { // 到內部時先判斷鄰居是不是外部，只要有一個是外部就視為外部
                    int dir;
                    for(dir = 0; dir < 6; dir++) {
                        int x = i + offsets[dir][0];
                        int y = j + offsets[dir][1];
                        int z = k + offsets[dir][2];
                        if(isRangeValid(x, y, z) && voxel_temp2[idx(x, y, z)] < 0) {
                            break;
                        }
                    }
                    if(dir == 6) 
                        continue; // 如果鄰居都不是外部才確定自己是內部，保持正值，繼續往內走
                }
                // 在外部就count++並距離值變負的
                voxel_out_count[idx(i, j, k)]++;
                voxel_temp2[idx(i, j, k)] = -voxel_temp2[idx(i, j, k)];
            }
        }
    }

    voxel_temp1 = voxel_distance, voxel_temp2 = voxel_distance;
    // 做z方向的判斷
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; j++) {
            // 從z=0開始往內走
            bool is_out = true;
            for(int k = 0; k < depth; k++) {
                  // 遇到模型就切換
                if(voxel_distance[idx(i, j, k)] == 0 && is_out == true) { // 第一次碰到模型才切換
                    is_out = !is_out;
                    continue;
                } else if(voxel_distance[idx(i, j, k)] == 0) continue; // 第二次以後碰到模型直接跳過

                if(!is_out) { // 到內部時先判斷鄰居是不是外部，只要有一個是外部就視為外部
                    int dir;
                    for(dir = 0; dir < 6; dir++) {
                        int x = i + offsets[dir][0];
                        int y = j + offsets[dir][1];
                        int z = k + offsets[dir][2];
                        if(isRangeValid(x, y, z) && voxel_temp1[idx(x, y, z)] < 0) {
                            break;
                        }
                    }
                    if(dir == 6) 
                        continue; // 如果鄰居都不是外部才確定自己是內部，保持正值，繼續往內走
                }
                // 在外部就count++並距離值變負的
                voxel_out_count[idx(i, j, k)]++;
                voxel_temp1[idx(i, j, k)] = -voxel_temp1[idx(i, j, k)];
            }
        }
        for(int j = height - 1; j >= 0; j--) {
            // 從z=depth - 1開始往內走
            bool is_out = true;
            for(int k = depth - 1; k >= 0; k--) {
                  // 遇到模型就切換
                if(voxel_distance[idx(i, j, k)] == 0 && is_out == true) { // 第一次碰到模型才切換
                    is_out = !is_out;
                    continue;
                } else if(voxel_distance[idx(i, j, k)] == 0) continue; // 第二次以後碰到模型直接跳過

                if(!is_out) { // 到內部時先判斷鄰居是不是外部，只要有一個是外部就視為外部
                    int dir;
                    for(dir = 0; dir < 6; dir++) {
                        int x = i + offsets[dir][0];
                        int y = j + offsets[dir][1];
                        int z = k + offsets[dir][2];
                        if(isRangeValid(x, y, z) && voxel_temp2[idx(x, y, z)] < 0) {
                            break;
                        }
                    }
                    if(dir == 6) 
                        continue; // 如果鄰居都不是外部才確定自己是內部，保持正值，繼續往內走
                }
                // 在外部就count++並距離值變負的
                voxel_out_count[idx(i, j, k)]++;
                voxel_temp2[idx(i, j, k)] = -voxel_temp2[idx(i, j, k)];
            }
        }
    }

    // 最後累積四個以上被判斷為外部才是外部，將距離值改為負的
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; j++) {
            for(int k = 0; k < depth; k++) {
                if(voxel_out_count[idx(i, j, k)] >= 3) {
                    voxel_distance[idx(i, j, k)] = -voxel_distance[idx(i, j, k)];
                }
            }
        }
    }
}