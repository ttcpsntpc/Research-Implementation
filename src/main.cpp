#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "Object.h"
#include "Shader.h"
#include "vertex.h"
#include "texture.h"
#include "camera.h"
#include "read_file.h"
#include "UIManager.h"
#include "distance_field_3D.h"

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <math.h>
#include <vector>
#include <algorithm>
#include <thread>
#include <random>

using namespace std;

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods);
void processInput(GLFWwindow *window);

// extra function
vector<glm::vec3> extractModelSurface(bool isSOM, VolumeData &df_data, vector<Vertex_c> &vertex); 

// settings
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

// Camera
Camera_c camera(glm::vec3(100.0f, 100.0f, 500.0f));
glm::vec3 lightPos(500.0f, 500.0f, 500.0f);
bool moveObject = 0; // 在移動光源或是相機

// timing
float deltaTime = 0.0f; // time between current frame and last frame
float lastFrame = 0.0f;

vector<Vertex_c> vertices_tri;
// (1) 讓BB對齊學姊DF的大小
// VolumeData volume_data = ReadObjFile("../../raw/teapot.obj", &vertices_tri); 
DistanceField3D DF;
UIManager UI;

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // GL 3.0 + GLSL 130
    const char *glsl_version = "#version 130";
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;

    // Setup Dear ImGui style  
    ImGui::StyleColorsDark();
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    Shader_c shader("shader/shader.vs", "shader/shader.fs");
    Shader_c light_shader("shader/light_shader.vs", "shader/light_shader.fs");

    vertex.CreateVertices();
    Object_c square_red;
    square_red.CreateObject(vertices[0], {});
    Object_c square_blue;
    square_blue.CreateObject(vertices[1], {});
    Object_c cube;
    cube.CreateObject(vertices[2], {});
    Object_c axis;  
    axis.CreateObject(vertices[3], {});
    Object_c light_cube;
    light_cube.CreateObject(vertices[4], {});
    Object_c Object_tri;
    Object_tri.CreateObject(vertices_tri, {});

    // (1)
    // Object_c object_bi_voxel;
    // object_bi_voxel.CreateObject(vertices[5], {});

    // vector<glm::vec3> cube_pos; 
    // for(int i = 0; i < volume_data.resolution[0]; i++)
    //     for(int j = 0; j < volume_data.resolution[1]; j++)
    //         for(int k = 0; k < volume_data.resolution[2]; k++) 
    //             if(volume_data(i, j, k)==255)
    //                 cube_pos.push_back({i, j, k});
    // cout<<"Model voxel size: "<<cube_pos.size()<<endl;
    // object_bi_voxel.UpdateInstances(cube_pos);

    Object_c object_df_voxel;
    object_df_voxel.CreateObject(vertices[5], {});

    VolumeData df_data = ReadVolumeFile("../../raw/df_data/teapot_dist.inf", "../../raw/df_data/teapot_df.raw", false); // 學姊的
    // 取得體素表面頂點
    vector<Vertex_c> voxel_vertex;
    extractModelSurface(false, df_data, voxel_vertex);
    Object_c object_model;
    object_model.CreateObject(voxel_vertex, {});

    UI.init();

    // 顯示時統一縮放到一定範圍
    // 將(解析度 * voxel size)最高的那側作為分母縮放到512大小 並乘以voxel size
    double max_range = max(df_data.resolution[0] * df_data.voxel_size[0], max(df_data.resolution[1] * df_data.voxel_size[1], df_data.resolution[2] * df_data.voxel_size[2]));
    glm::vec3 ratio(512 / max_range); 
    ratio *= df_data.voxel_size;
    int distance_range = 0;


    bool is_draw_bi_voxels = false, is_draw_df_voxels = false, is_draw_model_triangles = false, is_draw_bounding_box = true;

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui::NewFrame();
        UI.render(lightPos, camera.Position);
        ImGui::Begin("GUI");
        ImGui::SliderInt("showing distance", &distance_range, ((int)df_data.minimum - 1), (int)df_data.maximum + 1);
        if(ImGui::Button("update voxels")) {
            vector<glm::vec3> cube_pos;
            for(int i = 0; i < df_data.resolution[0]; i++)
                for(int j = 0; j < df_data.resolution[1]; j++)
                    for(int k = 0; k < df_data.resolution[2]; k++) 
                        if(df_data(i, j, k) > distance_range - 0.5 && df_data(i, j, k) < distance_range + 0.5)
                            cube_pos.push_back({i, j, k});
            cout<<"distance: "<<distance_range<<"  voxel number: "<<cube_pos.size()<<endl;
            object_df_voxel.UpdateInstances(cube_pos);
        }
        ImGui::Checkbox("draw model triangles", &is_draw_model_triangles); //(1)
        // ImGui::Checkbox("draw model bi-voxels", &is_draw_bi_voxels);
        ImGui::Checkbox("draw model df-voxels", &is_draw_df_voxels);
        ImGui::Checkbox("draw model bounding box", &is_draw_bounding_box);
        
        
        struct GridPoint {
    int x, y;
    // 讓 vector 可以去重或搜尋
    bool operator==(const GridPoint& other) const { return x == other.x && y == other.y; }
};

// 模擬全域或類別成員變數，用來儲存已選取的點
static std::vector<GridPoint> selected_points;

// 網格設定
const float GRID_SIZE = 40.0f;       // 每個網格的大小（像素）
const float POINT_RADIUS = 6.0f;     // 節點點擊視覺反饋的半徑
const int GRID_COLS = 15;            // 網格總欄數
const int GRID_ROWS = 15;            // 網格總列數

// 開始 ImGui 視窗
ImGui::Begin("互動網格畫布");

// 1. 定義一個隱形的自訂元件作為畫布，用來捕捉滑鼠事件
ImVec2 canvas_pos = ImGui::GetCursorScreenPos();       // 畫布左上角在螢幕上的絕對坐標
ImVec2 canvas_size(GRID_COLS * GRID_SIZE, GRID_ROWS * GRID_SIZE); // 畫布總大小
ImGui::InvisibleButton("canvas", canvas_size);          // 建立一個隱形按鈕佔位

// 取得當前視窗的繪圖列表
ImDrawList* draw_list = ImGui::GetWindowDrawList();

// 2. 繪製網格背景線
ImU32 grid_color = IM_COL32(200, 200, 200, 40); // 淺灰色，帶透明度
for (int i = 0; i <= GRID_COLS; i++) {
    float x = canvas_pos.x + i * GRID_SIZE;
    draw_list->AddLine(ImVec2(x, canvas_pos.x), ImVec2(x, canvas_pos.y + canvas_size.y), grid_color);
}
for (int j = 0; j <= GRID_ROWS; j++) {
    float y = canvas_pos.y + j * GRID_SIZE;
    draw_list->AddLine(ImVec2(canvas_pos.x, y), ImVec2(canvas_pos.x + canvas_size.x, y), grid_color);
}

// 3. 處理滑鼠點擊偵測
if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
{
    ImVec2 mouse_pos = ImGui::GetMousePos();
    
    // 計算滑鼠相對於畫布左上角的偏移量
    float local_x = mouse_pos.x - canvas_pos.x;
    float local_y = mouse_pos.y - canvas_pos.y;

    // 四捨五入到最近的網格交叉點坐標 (Grid Coordinates)
    int grid_x = (int)std::round(local_x / GRID_SIZE);
    int grid_y = (int)std::round(local_y / GRID_SIZE);

    // 確保在合法網格範圍內
    if (grid_x >= 0 && grid_x <= GRID_COLS && grid_y >= 0 && grid_y <= GRID_ROWS)
    {
        GridPoint clicked_p = { grid_x, grid_y };
        
        // 檢查是否已經選取過：若有則取消選取（Toggle），若無則加入
        auto it = std::find(selected_points.begin(), selected_points.end(), clicked_p);
        if (it != selected_points.end()) {
            selected_points.erase(it); // 取消選取
        } else {
            selected_points.push_back(clicked_p); // 新增選取
        }
    }
}

// 4. 繪製所有已被選取的點
ImU32 select_color = IM_COL32(255, 99, 71, 255); // 鮮豔的番茄紅
for (const auto& p : selected_points)
{
    // 將網格坐標還原為螢幕絕對像素坐標
    float screen_x = canvas_pos.x + p.x * GRID_SIZE;
    float screen_y = canvas_pos.y + p.y * GRID_SIZE;
    
    // 在交叉點上畫一個實心圓形
    draw_list->AddCircleFilled(ImVec2(screen_x, screen_y), POINT_RADIUS, select_color);
}

ImGui::End();




        ImGui::End();

        // create model matrix
        glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        // create view matrix
        glm::mat4 view = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        // create projection matrix
        glm::mat4 projection = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first

        // active shader for shading
        shader.use();

        view = camera.GetViewMatrix();
        projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 10000.0f);
        shader.setMat4("view", view);
        shader.setMat4("projection", projection);
        shader.setVec3("lightPos", lightPos);
        shader.setVec3("viewPos", camera.Position);

        // 畫一個測試方塊
        // glBindVertexArray(cube.VAO_);
        // model = glm::mat4(1.0f);
        // model = glm::translate(model, glm::vec3(10.0f, 10.0f, 10.0f));
        // model = glm::scale(model, glm::vec3(10, 10, 10));
        // shader.setMat4("model", model);
        // glDrawArrays(GL_TRIANGLES, 0, cube.size);
        
        // (1)
        // 畫模型的三角片
        if(is_draw_model_triangles) {
            glBindVertexArray(object_model.VAO_);
            model = glm::mat4(1.0f); 
            model = glm::scale(model, ratio);
            shader.setMat4("model", model);
            shader.setVec3("color", glm::vec3(1.0, 0.5, 0.0));
            glDrawArrays(GL_TRIANGLES, 0, object_model.size);
        }

        // 畫binary voxels
        // if(is_draw_bi_voxels) {
        //     glBindVertexArray(object_bi_voxel.VAO_);
        //     model = glm::mat4(1.0f);
        //     model = glm::scale(model, glm::vec3(volume_data.voxel_size[0], volume_data.voxel_size[1], volume_data.voxel_size[2])); // 因為存位置時是直接用index存，需轉換成實際體素大小
        //     model = glm::scale(model, ratio);
        //     light_shader.setMat4("model", model);
        //     glDrawArraysInstanced(GL_TRIANGLES, 0, object_bi_voxel.size, object_bi_voxel.instanceCount); 
        // }

        if(is_draw_df_voxels) {
            glBindVertexArray(object_df_voxel.VAO_);
            model = glm::mat4(1.0f);
            model = glm::scale(model, ratio);
            shader.setMat4("model", model);
            shader.setVec3("color", glm::vec3(1.0, 0.5, 0.0));
            glDrawArraysInstanced(GL_TRIANGLES, 0, object_df_voxel.size, object_df_voxel.instanceCount); 
        }


        // active shader for background
        light_shader.use();
        light_shader.setVec3("lightPos", lightPos);
        light_shader.setMat4("view", view);
        light_shader.setMat4("projection", projection);
        light_shader.setFloat("opacity", 1.0f);

        // draw the light
        glBindVertexArray(light_cube.VAO_);
        model = glm::mat4(1.0f);
        model = glm::translate(model, lightPos);
        model = glm::scale(model, glm::vec3(10, 10, 10));
        light_shader.setMat4("model", model);
        glDrawArrays(GL_TRIANGLES, 0, light_cube.size);
        
        // draw the 3 axis
        glBindVertexArray(axis.VAO_);
        model = glm::mat4(1.0f);
        model = glm::scale(model, glm::vec3(100, 100, 100));
        light_shader.setMat4("model", model);
        glDrawArrays(GL_LINES, 0, axis.size);
        
        // 畫 bounding box
        if(is_draw_bounding_box) {
            glDepthMask(GL_FALSE); 
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_CULL_FACE);
            glCullFace(GL_FRONT);
            light_shader.setFloat("opacity", 0.5f);
            glBindVertexArray(cube.VAO_);
            model = glm::mat4(1.0f);
            model = glm::scale(model, glm::vec3(df_data.resolution[0], df_data.resolution[1], df_data.resolution[2]));
            model = glm::scale(model, ratio);
            light_shader.setMat4("model", model);
            glDrawArrays(GL_TRIANGLES, 0, cube.size);
            glCullFace(GL_BACK);
            light_shader.setFloat("opacity", 0.1f);
            glDrawArrays(GL_TRIANGLES, 0, cube.size);
            glDisable(GL_CULL_FACE);
            glDepthMask(GL_TRUE);
        }
        
        // ImGui render
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
        moveObject = !moveObject;
    }
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if(moveObject == 0)
    {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FRONT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            camera.ProcessKeyboard(BACK, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            camera.ProcessKeyboard(LEFT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            camera.ProcessKeyboard(RIGHT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            camera.ProcessKeyboard(UP, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
            camera.ProcessKeyboard(DOWN, deltaTime);
    }
    else
    {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            lightPos.z -= 100.0f * deltaTime; 
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            lightPos.z += 100.0f * deltaTime; 
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            lightPos.x -= 100.0f * deltaTime; 
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            lightPos.x += 100.0f * deltaTime; 
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            lightPos.y += 100.0f * deltaTime; 
        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
            lightPos.y -= 100.0f * deltaTime; 
    }

    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
        camera.ProcessKeyboard(PITCHUP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
        camera.ProcessKeyboard(PITCHDOWN, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
        camera.ProcessKeyboard(YAWLEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
        camera.ProcessKeyboard(YAWRIGHT, deltaTime);

}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// 找模型表面體素，作為SOM的input並取得繪製體素的表面
vector<glm::vec3> extractModelSurface(bool isSOM, VolumeData &df_data, vector<Vertex_c> &vertex) {
    vector<glm::vec3> input_vector;
    
    // 建立一個 lambda 輔助函式，用來插入頂點並自動加上位移
    auto insertAndOffsetFace = [&](const vector<Vertex_c>& face_vertices, int x_offset, int y_offset, int z_offset) {
        size_t old_size = vertex.size();
        vertex.insert(vertex.end(), face_vertices.begin(), face_vertices.end());
        size_t new_size = vertex.size();
        
        for(size_t v = old_size; v < new_size; v++) {
            vertex[v].position.x += x_offset;
            vertex[v].position.y += y_offset;
            vertex[v].position.z += z_offset;
        }
    };

    for(int i = 0; i < df_data.resolution[0]; i++) {
        for(int j = 0; j < df_data.resolution[1]; j++) {
            for(int k = 0; k < df_data.resolution[2]; k++) {
                if(df_data(i, j, k) == 0) { // 遇到model檢查鄰居是否是空氣
                    bool isInput = false;
                    if((i == 0) || (df_data(i - 1, j, k)) != 0) { // -x
                        insertAndOffsetFace(cube_face_vertices[0], i, j, k);
                        isInput = true;
                    }
                    if((i == df_data.resolution[0] - 1) || (df_data(i + 1, j, k)) != 0) { // +x
                        insertAndOffsetFace(cube_face_vertices[1], i, j, k);
                        isInput = true;
                    }
                    if((j == 0) || (df_data(i, j - 1, k)) != 0) { // -y
                        insertAndOffsetFace(cube_face_vertices[2], i, j, k);
                        isInput = true;
                    }
                    if((j == df_data.resolution[1] - 1) || (df_data(i, j + 1, k)) != 0) { // +y
                        insertAndOffsetFace(cube_face_vertices[3], i, j, k);
                        isInput = true;
                    }
                    if((k == 0) || (df_data(i, j, k - 1)) != 0) { // -z
                        insertAndOffsetFace(cube_face_vertices[4], i, j, k);
                        isInput = true;
                    }
                    if((k == df_data.resolution[2] - 1) || (df_data(i, j, k + 1)) != 0) { // +z
                        insertAndOffsetFace(cube_face_vertices[5], i, j, k);
                        isInput = true;
                    }

                    if(isSOM && isInput)
                        input_vector.push_back(glm::vec3(i, j, k));
                }
            }
        }
    }
    return input_vector;
}