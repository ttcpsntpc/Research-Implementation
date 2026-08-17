#include <chrono>
#include "read_file.h"
#include "tri_box_overlap.h"

using namespace chrono; // for 計算時間差

VecData ReadVecFile(const char *vec_filename)
{
    VecData vec_file;

    ifstream ifs(vec_filename, ios::in);
    if(ifs.fail())
    {
        cout<< "Failed to open vec file." << '\n';
        return vec_file;
    }

    if (!(ifs >> vec_file.resolution[0] >> vec_file.resolution[1])) {
        cout << "Failed to read resolution." << endl;
        return vec_file;
    }

    auto start = steady_clock::now();

    int total_vector = vec_file.resolution[0] * vec_file.resolution[1];
    vec_file.data.clear();
    vec_file.data.reserve(total_vector);

    float max_speed = 0;
    float x, y;
    for(int i = 0; i < total_vector; i++) {
        if(ifs >> y >> x) {
            vec_file.data.push_back(glm::vec2(x, y));
            float speed = sqrt(x * x + y * y);
            max_speed = (speed > max_speed) ? speed : max_speed;
        }
        else {
            cout<<"Warning: 資料實際個數和resolution乘積不一致"<<endl;
            return vec_file;
        }
    }
    vec_file.max_speed = max_speed;

    ifs.close();

    auto end = steady_clock::now();
    auto diff = duration_cast<milliseconds>(end - start);
    cout<<"讀vec檔時間: "<<diff.count()<<" ms"<<endl;

    cout<<"vector resolution: "<<vec_file.resolution[0]<<" "<<vec_file.resolution[1]<<endl;
    cout<<"max speed: "<<vec_file.max_speed<<endl;
    return vec_file;
}

bool ReadInfFile(const char *inf_filename, VolumeData &volume_data)
{
    ifstream ifs(inf_filename, ios::in);
    if (ifs.fail())
    {
        cout << "Failed to open inf file." << '\n';
        return false;
    }

    auto start = steady_clock::now();

    string line;
    //string raw_file_name;
    while(getline(ifs, line))
    {
        if(line.empty() || line[0] == '#') continue; // 跳過空行和註解行
        size_t pos = line.find('=');
        if (pos == string::npos) continue; // 沒有等號的行也跳過

        // key跟value改成全部小寫，以免大小寫不一致導致讀取失敗
        string key = line.substr(0, pos);
        string value = line.substr(pos + 1);
        transform(key.begin(), key.end(), key.begin(), ::tolower);
        transform(value.begin(), value.end(), value.begin(), ::tolower);
        
        if(key == "raw-file" || key == "filename")   
        {
            //raw_file_name = value; 
        }
        // resolution的格式不固定，中間用%*c直接跳過
        else if(key.find("resolution") != string::npos)
            sscanf(value.c_str(), "%d%*c%d%*c%d", 
                &volume_data.resolution[0], 
                &volume_data.resolution[1], 
                &volume_data.resolution[2]);
        else if(key.find("sample") != string::npos || key.find("value") != string::npos) {
            if(value.find("int") != string::npos)
                volume_data.sample_type = INT;
            else if(value.find("unsigned") != string::npos) {
                if(value.find("int") != string::npos)
                    volume_data.sample_type = UNSIGNED_INT;
                else if(value.find("char") != string::npos)
                    volume_data.sample_type = UNSIGNED_CHAR;
                else if(value.find("short") != string::npos)
                    volume_data.sample_type = UNSIGNED_SHORT;
            }
            else if(value.find("char") != string::npos)
                volume_data.sample_type = CHAR;
            else if(value.find("float") != string::npos)
                volume_data.sample_type = FLOAT;
            else if(value.find("short") != string::npos)
                volume_data.sample_type = SHORT;
            // 完整型態名稱都沒有就代表是縮寫
            else if(value.find("ui") != string::npos)
                volume_data.sample_type = UNSIGNED_INT;
            else if(value.find("ub") != string::npos)
                volume_data.sample_type = UNSIGNED_CHAR;
            else if(value.find("us") != string::npos)
                volume_data.sample_type = UNSIGNED_SHORT;
            else if(value.find("i") != string::npos)
                volume_data.sample_type = INT;
            else if(value.find("b") != string::npos)
                volume_data.sample_type = CHAR;
            else if(value.find("f") != string::npos)
                volume_data.sample_type = FLOAT;
            else if(value.find("s") != string::npos)
                volume_data.sample_type = SHORT;
        }
        else if(key.find("voxel-size") != string::npos)
            sscanf(value.c_str(), "%f%*c%f%*c%f", 
                &volume_data.voxel_size[0], 
                &volume_data.voxel_size[1], 
                &volume_data.voxel_size[2]);
        else if(key.find("endian") != string::npos) {
            if(value.find("b") != string::npos) 
                volume_data.endian = BIG;
            else 
                volume_data.endian = LITTLE;
        }
        else if(key.find("max") != string::npos)
            sscanf(value.c_str(), "%f%*c%f%*c%f", 
                &volume_data.max.x, 
                &volume_data.max.y, 
                &volume_data.max.z);
        else if(key.find("min") != string::npos)
            sscanf(value.c_str(), "%f%*c%f%*c%f", 
                &volume_data.min.x, 
                &volume_data.min.y, 
                &volume_data.min.z);
    }
    
    auto end = steady_clock::now();
    auto diff = duration_cast<milliseconds>(end - start);
    cout<<"讀inf檔時間: "<<diff.count()<<" ms"<<endl;

    return true;
}

template <typename T>
T ReadAndConvert(EndianType endian, std::ifstream& ifs) {
    T value;
    char buffer[sizeof(T)];
    ifs.read(buffer, sizeof(T));

    // 預設資料是 Little-Endian
    // 如果檔案宣告為 BIG Endian，就將讀到的 Byte 順序反轉
    if (endian == BIG) {
        std::reverse(buffer, buffer + sizeof(T));
    }

    // 用C++ 的 memcpy 處理二進位轉型較安全
    std::memcpy(&value, buffer, sizeof(T));
    return value;
}

bool ReadRawFile(const char *raw_filename, VolumeData &volume_data)
{
    if(raw_filename == nullptr)
    {
        return false;
    }
    ifstream ifs(raw_filename, ios::in | ios::binary);
    if (ifs.fail())
    {
        cout << "Failed to open raw file." << '\n';
        return false;
    }

    auto start = steady_clock::now();
    
    int x = volume_data.resolution[0];
    int y = volume_data.resolution[1];
    int z = volume_data.resolution[2];

    volume_data.size = x * y * z; 

    // 讀進的數值都先用float存
    vector<float> temp_float_data;
    temp_float_data.clear();
    temp_float_data.reserve(volume_data.size);
    volume_data.voxel_data.reserve(volume_data.size);

    float minimum = numeric_limits<float>::max();
    float maximum = numeric_limits<float>::lowest();

    // 根據不同inf的資訊決定怎麼讀並算最大最小值
    for(int i = 0; i < volume_data.size; i++) {
        float value;

        switch(volume_data.sample_type) {
            case CHAR:           value = static_cast<float>(ReadAndConvert<char>(volume_data.endian, ifs)); break;
            case UNSIGNED_CHAR:  value = static_cast<float>(ReadAndConvert<unsigned char>(volume_data.endian, ifs)); break;
            case SHORT:          value = static_cast<float>(ReadAndConvert<short>(volume_data.endian, ifs)); break;
            case UNSIGNED_SHORT: value = static_cast<float>(ReadAndConvert<unsigned short>(volume_data.endian, ifs)); break;
            case INT:            value = static_cast<float>(ReadAndConvert<int>(volume_data.endian, ifs)); break;
            case UNSIGNED_INT:   value = static_cast<float>(ReadAndConvert<unsigned int>(volume_data.endian, ifs)); break;
            case FLOAT:          value = ReadAndConvert<float>(volume_data.endian, ifs); break;
        }

        if(value > maximum) maximum = value;
        if(value < minimum) minimum = value;

        temp_float_data.push_back(value);
    }
    cout<<endl<<"maximum: "<<maximum<<" minimum: "<<minimum<<endl;

    volume_data.voxel_data.clear();

    // 將資料正規化到0~255
    float range = maximum - minimum;
    for(int i = 0; i < volume_data.size; i++) {
        float normalized_data = ((temp_float_data[i] - minimum) / range) * 255.0f;
        if(normalized_data < 0 || normalized_data > 255)
            cout<<"錯了";
        unsigned char data = static_cast<unsigned char>(normalized_data + 0.5f);
        volume_data.voxel_data.push_back(data);
        volume_data.intensity_counts[data]++;
    }
    
    auto end = steady_clock::now();
    auto diff = duration_cast<milliseconds>(end - start);
    cout<<"讀raw檔時間: "<<diff.count()<<" ms"<<endl;

    return true;
}

VolumeData ReadVolumeFile(const char *inf_filename, const char *raw_filename) {
    VolumeData volume_data;
    if (!ReadInfFile(inf_filename, volume_data)) {
        cout<<"Reading inf file error"<<endl;
    }
    if (!ReadRawFile(raw_filename, volume_data)) {
        cout<<"Reading raw file error"<<endl;
    }
    return volume_data;
}

// 最基本的資料平均方法
vector<unsigned char> cumulativeDistributionEqualization(VolumeData volume_data) {
    int width = volume_data.resolution[0];
    int height = volume_data.resolution[1];
    int depth = volume_data.resolution[2];
    vector<unsigned char> equalized_data; 
    equalized_data.resize(width * height * depth);
    int count[256] = {0}, sumCount[256] = {0};

    for(int k = 0; k < depth; k++)
        for(int j = 0; j < height; j++)
            for(int i = 0; i < width; i++)
                count[volume_data(i, j, k)]++;
    
    for(int i = 1; i < 256; i++)
        sumCount[i] += sumCount[i - 1] + count[i - 1];
    for(int i = 1; i < 256; i++)
        sumCount[i] = sumCount[i] / (float)volume_data.size * 255.0f + 0.4f;
        
    for(int k = 0; k < depth; k++)
        for(int j = 0; j < height; j++)
            for(int i = 0; i < width; i++)
                equalized_data[volume_data.idx(i, j, k)] = sumCount[volume_data(i, j, k)];
    cout<<"CDE finished！"<<endl;
    return equalized_data;
}

// 用積分圖儲存每個體素累積起來的各個硬度值個數，更新硬度值時直接從積分圖取值做運算即可，都是以目前體素當作中心擴展直方圖統計範圍，因此遇邊緣時滑動視窗會被刪減
// 需注意記憶體大小 Ex: 256*256*256*256*4bytes ~= 17.18GB
vector<unsigned char> adaptiveHistogramEqualization(VolumeData volume_data, int side) { // side代表正方體的邊長，需設成奇數
    int offsets = side / 2;
    int width = volume_data.resolution[0];
    int height = volume_data.resolution[1];
    int depth = volume_data.resolution[2];

    vector<unsigned char> equalized_data;
    equalized_data.resize(width * height * depth);

    vector<vector<int>> integral_data;
    integral_data.resize(width * height * depth, vector<int>(256, 0));
    
    // 1-1: 初始化，將每個 voxel 的原始數值存入對應的 bin
    for(int k = 0; k < depth; k++) {
        for(int j = 0; j < height; j++) {
            for(int i = 0; i < width; i++) {
                int val = volume_data(i, j, k);
                integral_data[volume_data.idx(i, j, k)][val] = 1;
            }
        }
    }

    // 對 X 軸累加
    for(int k = 0; k < depth; k++) {
        for(int j = 0; j < height; j++) {
            for(int i = 1; i < width; i++) {
                for(int b = 0; b < 256; b++) {
                    integral_data[volume_data.idx(i, j, k)][b] += integral_data[volume_data.idx(i - 1, j, k)][b];
                }
            }
        }
    }
    // 對 Y 軸累加
    for(int k = 0; k < depth; k++) {
        for(int j = 1; j < height; j++) {
            for(int i = 0; i < width; i++) {
                for(int b = 0; b < 256; b++) {
                    integral_data[volume_data.idx(i, j, k)][b] += integral_data[volume_data.idx(i, j - 1, k)][b];
                }
            }
        }
    }
    // 對 Z 軸累加
    for(int k = 1; k < depth; k++) {
        for(int j = 0; j < height; j++) {
            for(int i = 0; i < width; i++) {
                for(int b = 0; b < 256; b++) {
                    integral_data[volume_data.idx(i, j, k)][b] += integral_data[volume_data.idx(i, j, k - 1)][b];
                }
            }
        }
    }

    // 輔助 Lambda 函式：安全讀取積分圖，超出邊界時自動截斷，避免 Out of bounds
    auto getIntegral = [&](int x, int y, int z, int b) -> int {
        x = std::max(0, std::min(x, width - 1));
        y = std::max(0, std::min(y, height - 1));
        z = std::max(0, std::min(z, depth - 1));
        return integral_data[volume_data.idx(x, y, z)][b];
    };

    // 利用積分圖進行局部直方圖等化
    for(int k = 0; k < depth; k++) {
        for(int j = 0; j < height; j++) {
            for(int i = 0; i < width; i++) {
                
                // 計算局部視窗的 XYZ 範圍 (包含邊界外推)
                int x_max = i + offsets;
                int y_max = j + offsets;
                int z_max = k + offsets;
                
                // 積分圖的特性，計算下限時需要多減去 1
                int x_min = i - offsets - 1;
                int y_min = j - offsets - 1;
                int z_min = k - offsets - 1;

                int target_val = volume_data(i, j, k);
                int sumCount = 0;

                for(int b = 0; b <= target_val; b++) {
                    int count_in_window = 
                          getIntegral(x_max, y_max, z_max, b)
                        - getIntegral(x_min, y_max, z_max, b)
                        - getIntegral(x_max, y_min, z_max, b)
                        - getIntegral(x_max, y_max, z_min, b)
                        + getIntegral(x_min, y_min, z_max, b)
                        + getIntegral(x_min, y_max, z_min, b)
                        + getIntegral(x_max, y_min, z_min, b)
                        - getIntegral(x_min, y_min, z_min, b);
                    
                    sumCount += count_in_window;
                }

                // 計算該視窗實際涵蓋的 Voxel 總數 (需考量邊界處視窗會被截斷)
                int actual_x_min = std::max(0, i - offsets);
                int actual_x_max = std::min(width - 1, i + offsets);
                int actual_y_min = std::max(0, j - offsets);
                int actual_y_max = std::min(height - 1, j + offsets);
                int actual_z_min = std::max(0, k - offsets);
                int actual_z_max = std::min(depth - 1, k + offsets);
                
                float windowVolume = (actual_x_max - actual_x_min + 1) * (actual_y_max - actual_y_min + 1) * (actual_z_max - actual_z_min + 1);

                // 計算新的灰階值
                float equalized_val = ((float)sumCount * 255.0f) / windowVolume;
                
                // 限制在 0~255 範圍內
                equalized_data[volume_data.idx(i, j, k)] = (unsigned char)std::max(0.0f, std::min(255.0f, equalized_val));
            }
        }
    }
    cout<<"AHE finished！"<<endl;
    return equalized_data;
}

vector<unsigned char> CLAHE(VolumeData volume_data, int block_edge, float alpha) {
    // 取得資料維度
    int width = volume_data.resolution[0];
    int height = volume_data.resolution[1];
    int depth = volume_data.resolution[2];

    int block_num[3] = {width / block_edge, height / block_edge, depth / block_edge}; // 幾個block
    
    int total_blocks = block_num[0] * block_num[1] * block_num[2];
    std::vector<int> tiles_cdf(total_blocks * 256, 0); // 紀錄每個block的cdf

    // 輔助函式：計算 4D index
    auto getTileidx = [&](int i, int j, int k, int bin) -> int {
        return ((k * block_num[1] * block_num[0]) + (j * block_num[0]) + i) * 256 + bin;
    };

    // 計算 Threshold
    int block_size = block_edge * block_edge * block_edge;
    int average = block_size / 256;
    int threshold = average * alpha;
    
    if (threshold < average) threshold = average;

    // 階段一：計算每個block的CDF
    for(int k = 0; k < block_num[2]; k++) {
        for(int j = 0; j < block_num[1]; j++) {
            for(int i = 0; i < block_num[0]; i++) {
                
                // 1. 統計該區塊的直方圖
                int tile_offset = getTileidx(i, j, k, 0); // 計算該區塊在cdf table的起始index
                // 計算該區塊在體素資料的起始index
                int start_x = i * block_edge;
                int start_y = j * block_edge;
                int start_z = k * block_edge;

                for(int z = start_z; z < start_z + block_edge; z++) {
                    for(int y = start_y; y < start_y + block_edge; y++) {
                        for(int x = start_x; x < start_x + block_edge; x++) {
                            int val = volume_data(x, y, z);
                            tiles_cdf[tile_offset + val]++;
                        }
                    }
                }

                // 2. 直方圖超出threshold的部分做Clipping
                bool needs_clipping = true;
                while (needs_clipping) {
                    int excess = 0;
                    
                    // 算多出多少
                    for (int b = 0; b < 256; b++) {
                        if (tiles_cdf[tile_offset + b] > threshold) {
                            excess += tiles_cdf[tile_offset + b] - threshold;
                            tiles_cdf[tile_offset + b] = threshold;
                        }
                    }

                    // 如果有多出來的，就重新分配
                    if (excess > 255) {
                        int step = excess / 256;
                        int rem = excess % 256;
                        
                        // 平均分配 Step
                        for (int b = 0; b < 256; b++) {
                            tiles_cdf[tile_offset + b] += step;
                        }
                        
                        // 分配餘數 (平均打散)
                        int b = 0;
                        while(rem > 0) {
                            tiles_cdf[tile_offset + b] += 1;
                            b++;
                            rem--;
                        }
                        // 重新跑一次 while 迴圈檢查是否有新的超出
                    } else {
                        // 沒有多出很多就跳出迴圈
                        needs_clipping = false; 
                    }
                }

                // 3. 計算累積分布函數 (CDF) 並轉換為 Mapping Table (LUT)
                int sum = 0;
                for (int b = 0; b < 256; b++) {
                    sum += tiles_cdf[tile_offset + b];
                    // 將 CDF 映射到 0~255 的灰階值
                    // 由於可能有些微精度誤差，強制約束在 0~255
                    int mapped_val = (sum * 255) / block_size;
                    tiles_cdf[tile_offset + b] = std::max(0, std::min(255, mapped_val));
                }
            }
        }
    }

    // 階段二：藉由鄰居跟映射表更新硬度值
    vector<unsigned char> equalized_data(width * height * depth);

    for(int z = 0; z < depth; z++) {
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                
                int val = volume_data(x, y, z);

                // 計算當前 Voxel 相對的「區塊小數座標」(找出它在哪些區塊中心之間)
                // -0.5 是因為區塊中心是在 block_edge 的一半
                float tx = ((float)x / block_edge) - 0.5f;
                float ty = ((float)y / block_edge) - 0.5f;
                float tz = ((float)z / block_edge) - 0.5f;

                // 找出左下前 (x1, y1, z1) 與 右上後 (x2, y2, z2) 的區塊 Index
                int x1 = std::max(0, (int)std::floor(tx));
                int y1 = std::max(0, (int)std::floor(ty));
                int z1 = std::max(0, (int)std::floor(tz));

                int x2 = std::min(block_num[0] - 1, x1 + 1);
                int y2 = std::min(block_num[1] - 1, y1 + 1);
                int z2 = std::min(block_num[2] - 1, z1 + 1);

                // 計算距離比例 (用於內插權重)
                float px = tx - std::floor(tx);
                float py = ty - std::floor(ty);
                float pz = tz - std::floor(tz);

                // 防呆：如果是邊緣的體素，權重設為 0，直接使用邊界區塊的數值
                if (x < block_edge / 2 || x >= width - block_edge / 2) px = 0.0f;
                if (y < block_edge / 2 || y >= height - block_edge / 2) py = 0.0f;
                if (z < block_edge / 2 || z >= depth - block_edge / 2) pz = 0.0f;

                // 從 8 個相鄰區塊的 CDF 中查出對應的轉換值
                float c000 = tiles_cdf[getTileidx(x1, y1, z1, val)];
                float c100 = tiles_cdf[getTileidx(x2, y1, z1, val)];
                float c010 = tiles_cdf[getTileidx(x1, y2, z1, val)];
                float c110 = tiles_cdf[getTileidx(x2, y2, z1, val)];
                float c001 = tiles_cdf[getTileidx(x1, y1, z2, val)];
                float c101 = tiles_cdf[getTileidx(x2, y1, z2, val)];
                float c011 = tiles_cdf[getTileidx(x1, y2, z2, val)];
                float c111 = tiles_cdf[getTileidx(x2, y2, z2, val)];

                // X 軸方向內插
                float c00 = c000 * (1 - px) + c100 * px;
                float c10 = c010 * (1 - px) + c110 * px;
                float c01 = c001 * (1 - px) + c101 * px;
                float c11 = c011 * (1 - px) + c111 * px;

                // Y 軸方向內插
                float c0 = c00 * (1 - py) + c10 * py;
                float c1 = c01 * (1 - py) + c11 * py;

                // Z 軸方向內插得到最終結果
                float final_val = c0 * (1 - pz) + c1 * pz;

                equalized_data[volume_data.idx(x, y, z)] = (unsigned char)std::max(0.0f, std::min(255.0f, final_val));
            }
        }
    }
    cout<<"CLAHE finished！"<<endl;
    return equalized_data;
}

TxtData ReadTxtFile(const char *txt_filename)
{
    TxtData txt_data;

    ifstream ifs(txt_filename, ios::in);
    if (ifs.fail())
    {
        cout << "Failed to open file." << '\n';
        return txt_data;
    }

    auto start = steady_clock::now();
    
    string my_line;

    getline(ifs, my_line, ' ');
    txt_data.resolution = stoi(my_line);
    getline(ifs, my_line, '\n');
    txt_data.dimension = stoi(my_line);

    glm::vec3 max_temp = glm::vec3(0.0f, 0.0f, 0.0f), min_temp = glm::vec3(9999.9f, 9999.9f, 9999.9f);
    for (int i = 0; i < txt_data.resolution; i++)
    {
        glm::vec3 temp;
        getline(ifs, my_line, ' ');
        temp.x = stof(my_line);
        if(temp.x > max_temp.x) max_temp.x = temp.x;
        else if (temp.x < min_temp.x) min_temp.x = temp.x;
        getline(ifs, my_line, ' ');
        temp.y = stof(my_line);
        if(temp.y > max_temp.y) max_temp.y = temp.y;
        else if (temp.y < min_temp.y) min_temp.y = temp.y;
        getline(ifs, my_line, ' ');
        temp.z = stof(my_line);
        if(temp.z > max_temp.z) max_temp.z = temp.z;
        else if (temp.z < min_temp.z) min_temp.z = temp.z;
        txt_data.weight.push_back(temp);
    }
    txt_data.max = max_temp;
    txt_data.min = min_temp;

    auto end = steady_clock::now();
    auto diff = duration_cast<milliseconds>(end - start);
    cout<<"讀檔時間: "<<diff.count()<<" ms"<<endl;

    return txt_data;
}

// SAT: 看看三角片跟六面體投影到哪個軸會不相交
void triangle2voxel(VolumeData &vol, glm::vec3 v[3]) {
    // 計算一個三角片的最大最小值
    glm::vec3 tri_min = glm::min(v[0], glm::min(v[1], v[2]));
    glm::vec3 tri_max = glm::max(v[0], glm::max(v[1], v[2]));

    glm::vec3 v_size(vol.voxel_size[0], vol.voxel_size[1], vol.voxel_size[2]); // 一個體素多長
    
    glm::vec3 h = v_size * 0.5f; // 半個體素多長

    // 將 Bounding Box 轉換成 Grid 的 Index 範圍 (並做邊界裁切)
    int min_i = std::max(0, static_cast<int>((tri_min.x - vol.min.x) / v_size.x));
    int min_j = std::max(0, static_cast<int>((tri_min.y - vol.min.y) / v_size.y));
    int min_k = std::max(0, static_cast<int>((tri_min.z - vol.min.z) / v_size.z));

    int max_i = std::min(vol.resolution[0] - 1, static_cast<int>((tri_max.x - vol.min.x) / v_size.x));
    int max_j = std::min(vol.resolution[1] - 1, static_cast<int>((tri_max.y - vol.min.y) / v_size.y));
    int max_k = std::min(vol.resolution[2] - 1, static_cast<int>((tri_max.z - vol.min.z) / v_size.z));

    // 輔助 Lambda 函式：用於 SAT 的一維投影測試 (可被編譯器完美 inline)
    // 如果分離 (Separated) 則回傳 true
    auto AxisTest = [](float p_a, float p_b, float rad) {
        return (std::max(p_a, p_b) < -rad || std::min(p_a, p_b) > rad);
    };

    // 只遍歷可能相交的局部 Voxel 範圍
    for (int k = min_k; k <= max_k; ++k) {
        for (int j = min_j; j <= max_j; ++j) {
            for (int i = min_i; i <= max_i; ++i) {
                
                // 計算當前 Voxel 的中心點座標
                glm::vec3 center;
                center.x = vol.min.x + (i + 0.5f) * v_size.x;
                center.y = vol.min.y + (j + 0.5f) * v_size.y;
                center.z = vol.min.z + (k + 0.5f) * v_size.z;

                // 將三角片的三個頂點平移，使 Voxel 中心成為原點 (0,0,0)
                glm::vec3 p0 = v[0] - center;
                glm::vec3 p1 = v[1] - center;
                glm::vec3 p2 = v[2] - center;

                // 計算三角片的三個邊向量
                glm::vec3 e0 = p1 - p0;
                glm::vec3 e1 = p2 - p1;
                glm::vec3 e2 = p0 - p2;

                // ==========================================================
                //  開始 Akenine-Möller SAT 測試 (總共 13 條分離軸)
                // ==========================================================

                // 【Test 1】 測試 3 條 AABB 面法向量 (X, Y, Z 軸)
                if (std::min({p0.x, p1.x, p2.x}) > h.x || std::max({p0.x, p1.x, p2.x}) < -h.x) continue;
                if (std::min({p0.y, p1.y, p2.y}) > h.y || std::max({p0.y, p1.y, p2.y}) < -h.y) continue;
                if (std::min({p0.z, p1.z, p2.z}) > h.z || std::max({p0.z, p1.z, p2.z}) < -h.z) continue;

                // 【Test 2】 測試 9 條交叉邊緣向量 (AABB 邊 x 三角片邊)
                // 由於外積時 e0 x p0 跟 e0 x p1是一樣的，所以每條軸只需要投影兩個點

                // AABB 軸 X (1,0,0) 與 三角邊 e0, e1, e2 的外積後 (變(0, -ei.z, ei.y))，再跟兩個頂點做投影
                if (AxisTest(p0.z * e0.y - p0.y * e0.z, p2.z * e0.y - p2.y * e0.z, h.y * std::abs(e0.z) + h.z * std::abs(e0.y))) continue;
                if (AxisTest(p0.z * e1.y - p0.y * e1.z, p1.z * e1.y - p1.y * e1.z, h.y * std::abs(e1.z) + h.z * std::abs(e1.y))) continue;
                if (AxisTest(p0.z * e2.y - p0.y * e2.z, p1.z * e2.y - p1.y * e2.z, h.y * std::abs(e2.z) + h.z * std::abs(e2.y))) continue;

                // AABB 軸 Y (0,1,0) 與 三角邊 e0, e1, e2 的外積
                if (AxisTest(-p0.z * e0.x + p0.x * e0.z, -p2.z * e0.x + p2.x * e0.z, h.x * std::abs(e0.z) + h.z * std::abs(e0.x))) continue;
                if (AxisTest(-p0.z * e1.x + p0.x * e1.z, -p1.z * e1.x + p1.x * e1.z, h.x * std::abs(e1.z) + h.z * std::abs(e1.x))) continue;
                if (AxisTest(-p0.z * e2.x + p0.x * e2.z, -p1.z * e2.x + p1.x * e2.z, h.x * std::abs(e2.z) + h.z * std::abs(e2.x))) continue;

                // AABB 軸 Z (0,0,1) 與 三角邊 e0, e1, e2 的外積
                if (AxisTest(p0.y * e0.x - p0.x * e0.y, p2.y * e0.x - p2.x * e0.y, h.x * std::abs(e0.y) + h.y * std::abs(e0.x))) continue;
                if (AxisTest(p0.y * e1.x - p0.x * e1.y, p1.y * e1.x - p1.x * e1.y, h.x * std::abs(e1.y) + h.y * std::abs(e1.x))) continue;
                if (AxisTest(p0.y * e2.x - p0.x * e2.y, p1.y * e2.x - p1.x * e2.y, h.x * std::abs(e2.y) + h.y * std::abs(e2.x))) continue;

                // 【Test 3】 測試 1 條三角片面法向量 (Plane vs AABB)
                glm::vec3 normal = glm::cross(e0, e1);
                // Voxel (AABB) 投影到法向量的半徑長度
                float r = h.x * std::abs(normal.x) + h.y * std::abs(normal.y) + h.z * std::abs(normal.z);
                // 檢查 Voxel 中心 (也就是當前原點) 到三角面平面的距離
                if (std::abs(glm::dot(normal, p0)) > r) continue; 

                //  13 條軸測試全數通過！判定為相交
                vol(i, j, k) = 255; 
            }
        }
    }
}

// 可簡單判斷STL檔案格式是binary或ASCII
VolumeData ReadStlFile(const char *stl_filename) {
    VolumeData stl_data;
    
    auto start = steady_clock::now();
    
    // 快速讀一次來判斷格式
    ifstream ifs_check(stl_filename, ios::in | ios::binary | ios::ate);
    if(!ifs_check.is_open()) {
        cout<<"Failed to open file."<<endl;
        return stl_data;
    }

    streamsize file_size = ifs_check.tellg(); // 檔案開頭到當下的byte數
    ifs_check.clear();
    ifs_check.seekg(80, ios::beg); // 從開頭移動80個位元組 (binary前80個表示header)
    uint32_t num_triangles = 0;
    ifs_check.read(reinterpret_cast<char *>(&num_triangles), sizeof(uint32_t)); // 接下來4個bytes表示模形三角形個數
    ifs_check.close();

    bool is_binary = (file_size == 84 + (num_triangles * 50)); // 用bytes個數來判斷格式方式

    ifstream ifs;
    // 讀第一遍: 找範圍 (最大最小)
    // * 要先讀因為體素化要根據voxel_size計算bounding box
    stl_data.min = glm::vec3(numeric_limits<float>::max());
    stl_data.max = glm::vec3(numeric_limits<float>::lowest());

    if(is_binary) {
        ifs.open(stl_filename, ios::in | ios::binary);
        ifs.seekg(84, ios::beg); // 跳過header

        for(int i = 0; i < num_triangles; i++) {
            ifs.seekg(12, ios::cur); // 從現在移動12位元組 跳過normal vector
            
            for(int j = 0; j < 3; j++) {
                float x, y, z;
                ifs.read(reinterpret_cast<char*>(&x), sizeof(float));
                ifs.read(reinterpret_cast<char*>(&y), sizeof(float));
                ifs.read(reinterpret_cast<char*>(&z), sizeof(float));
                
                stl_data.min = glm::min(stl_data.min, glm::vec3{x, y, z});
                stl_data.max = glm::max(stl_data.max, glm::vec3{x, y, z});
            }
            ifs.seekg(2, ios::cur);
        } 
    } else {
        ifs.open(stl_filename, ios::in);

        string token;
        while(ifs >> token) {
            if(token == "vertex") {
                float x, y, z;
                ifs >> x >> y >> z;

                stl_data.min = glm::min(stl_data.min, glm::vec3{x, y, z});
                stl_data.max = glm::max(stl_data.max, glm::vec3{x, y, z});
            }
        }
    }
    
    // 算總三角片範圍、邊界 (邊緣各預留1個單位)
    glm::vec3 range = stl_data.max - stl_data.min;
    for(int i = 0; i < 3; i++)
        stl_data.voxel_size[i] = range[i] / (stl_data.resolution[i] - 2);
    stl_data.min -= stl_data.voxel_size;
    stl_data.max += stl_data.voxel_size;
    stl_data.size = stl_data.resolution[0] * stl_data.resolution[1] * stl_data.resolution[2];
    stl_data.voxel_data.resize(stl_data.size, 0);

    ifs.clear(); // 清除 EOF (End of File) 狀態標記
    ifs.seekg(0, ios::beg); // 將讀取指標 (get pointer) 歸零回到檔案開頭
    // 讀第二遍: 體素化
    if(is_binary) {
        ifs.seekg(84, ios::beg);

        for(int i = 0; i < num_triangles; i++) {
            ifs.seekg(12, ios::cur);

            glm::vec3 vertex[3];
            for(int j = 0; j < 3; j++) {
                ifs.read(reinterpret_cast<char*>(&vertex[j].x), sizeof(float));
                ifs.read(reinterpret_cast<char*>(&vertex[j].y), sizeof(float));
                ifs.read(reinterpret_cast<char*>(&vertex[j].z), sizeof(float));
            }
            ifs.seekg(2, ios::cur);

            triangle2voxel(stl_data, vertex);

            glm::vec3 d0 = vertex[1] - vertex[0];
            glm::vec3 d1 = vertex[2] - vertex[1];
            glm::vec3 normal(d0.y*d1.z - d0.z*d1.y, d0.z*d1.x - d0.x*d1.z, d0.x*d1.y - d0.y*d1.x);
            for(int i = 0; i < 3; i++) {
                vertices_tri.push_back(Vertex_c{{vertex[i].x, vertex[i].y, vertex[i].z}, {1.0, 0.8, 0.0}, {}, {normal.x, normal.y, normal.z}});
            }
        }
    } else {
        string token;
        while (ifs >> token) {
            if (token == "facet") {
                // "facet normal nx ny nz"
                string normal_str;
                float nx, ny, nz;
                ifs >> normal_str >> nx >> ny >> nz;
                // "outer loop"
                string outer, loop;
                ifs >> outer >> loop;

                // 讀取 3 個頂點
                glm::vec3 vertex[3];
                for (int i = 0; i < 3; ++i) {
                    string vertex_str;
                    ifs >> vertex_str; // "vertex"
                    ifs >> vertex[i].x >> vertex[i].y >> vertex[i].z;
                }
                triangle2voxel(stl_data, vertex);
                
                glm::vec3 d0 = vertex[1] - vertex[0];
                glm::vec3 d1 = vertex[2] - vertex[1];
                glm::vec3 normal(d0.y*d1.z - d0.z*d1.y, d0.z*d1.x - d0.x*d1.z, d0.x*d1.y - d0.y*d1.x);
                for(int i = 0; i < 3; i++) {
                    vertices_tri.push_back(Vertex_c{{vertex[i].x, vertex[i].y, vertex[i].z}, {1.0, 0.8, 0.0}, {}, {normal.x, normal.y, normal.z}});
                }
            }
        }
    }
    
    ifs.close();

    auto end = steady_clock::now();
    auto diff = duration_cast<milliseconds>(end - start);

    cout << "STL File Format: " << (is_binary ? "Binary" : "ASCII") <<endl;
    cout<<"讀stl檔時間: "<<diff.count() / 1000.0<<"s"<<endl;

    cout << "Bounding Box Min: (" << stl_data.min.x << ", " << stl_data.min.y << ", " << stl_data.min.z << ")" << endl;
    cout << "Bounding Box Max: (" << stl_data.max.x << ", " << stl_data.max.y << ", " << stl_data.max.z << ")" << endl;
    
    return stl_data;
}

// 暫不處裡face有超過三個頂點的情況
VolumeData ReadObjFile(const char *stl_filename) {
    VolumeData obj_data;

    auto start = steady_clock::now();

    ifstream ifs(stl_filename, ios::in);

    if(!ifs.is_open()) {
        cout<<"Failed to open file."<<endl;
    }

    vector<glm::vec3> vertices;
    obj_data.min = glm::vec3(numeric_limits<float>::max());
    obj_data.max = glm::vec3(numeric_limits<float>::lowest());

    // 讀第一遍: 找範圍 (最大最小)
    string token;
    while(ifs >> token) {
        if(token == "v") {
            glm::vec3 v;
            ifs >> v.x >> v.y >> v.z;
            vertices.push_back(v);

            obj_data.min = glm::min(obj_data.min, v);
            obj_data.max = glm::max(obj_data.max, v);
        }
    }

    glm::vec3 range = obj_data.max - obj_data.min;
    for(int i = 0; i < 3; i++)
        obj_data.voxel_size[i] = range[i] / (obj_data.resolution[i] - 2);
    obj_data.min -= obj_data.voxel_size;
    obj_data.max += obj_data.voxel_size;
    obj_data.size = obj_data.resolution[0] * obj_data.resolution[1] * obj_data.resolution[2];
    obj_data.voxel_data.resize(obj_data.size, 0);

    ifs.clear();
    ifs.seekg(0, ios::beg);
    // 讀第二遍: 體素化
    while(ifs >> token) {
        if(token == "f") {
            string str_v[3]; // 存三個"v/vt/vn"
            ifs >> str_v[0] >> str_v[1] >> str_v[2];

            glm::vec3 vertex[3];
            for(int i = 0; i < 3; i++) {
                size_t pos = str_v[i].find('/');
                // obj頂點索引從1開始 所以存取陣列時會減一
                int idx;
                int num_vertices = vertices.size();
                if(pos != string::npos)
                    idx = (stoi(str_v[i].substr(0, pos)) - 1 + num_vertices) % num_vertices; // 為了兼容index -1情況 (從最後往前推)
                else
                    idx = (stoi(str_v[i]) - 1 + num_vertices) % num_vertices;
                vertex[i] = vertices[idx];
            }
            triangle2voxel(obj_data, vertex);
            
            glm::vec3 d0 = vertex[1] - vertex[0];
            glm::vec3 d1 = vertex[2] - vertex[1];
            glm::vec3 normal(d0.y*d1.z - d0.z*d1.y, d0.z*d1.x - d0.x*d1.z, d0.x*d1.y - d0.y*d1.x);
            for(int i = 0; i < 3; i++) {
                vertices_tri.push_back(Vertex_c{{vertex[i].x, vertex[i].y, vertex[i].z}, {1.0, 0.8, 0.0}, {}, {normal.x, normal.y, normal.z}});
            }
        }
    }

    ifs.close();

    auto end = steady_clock::now();
    auto diff = duration_cast<milliseconds>(end - start);
    
    cout<<"讀obj檔時間: "<<diff.count() / 1000.0<<"s"<<endl;

    cout << "Bounding Box Min: (" << obj_data.min.x << ", " << obj_data.min.y << ", " << obj_data.min.z << ")" << endl;
    cout << "Bounding Box Max: (" << obj_data.max.x << ", " << obj_data.max.y << ", " << obj_data.max.z << ")" << endl;

    return obj_data;
}