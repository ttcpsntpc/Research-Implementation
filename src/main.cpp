#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
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
#include "SOM.h"
#include "SOM_mapping.h"

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
#include <chrono>

using namespace std;
using namespace chrono;

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods);
void processInput(GLFWwindow *window);

// extra function
vector<Vertex_c> getModelVertices(VolumeData &df_data, int distance_value);
vector<Vertex_c> getLatticesVertices(SomData som_data);
vector<Vertex_c> getColorModelVertices(LayerVoxels &layer_voxels, VolumeData &df_data, glm::vec3 uncarved_color);
vector<LayerVoxels> getLayerVoxel(VolumeData &df_data, glm::mat4 matrix, int layer);
vector<vector<glm::vec3>> getTexture(const char *filename, unsigned int &texture);
void engravedCarving(LayerVoxels &layer_voxels, VolumeData &carved_df_data);
void embossedCarving(LayerVoxels &layer_voxels, VolumeData &carved_df_data);
void dilation(VolumeData &carved_df_data, int threshold);
void erosion(VolumeData &carved_df_data, int threshold);
void saveOpenGLScreenshot(const string& filename);

// settings
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

// Camera
Camera_c camera(glm::vec3(250.0f, 250.0f, 800.0f));
glm::vec3 lightPos(500.0f, 500.0f, 500.0f);
int moveObject = 0; // 在移動光源或是相機

// timing
float deltaTime = 0.0f; // time between current frame and last frame
float lastFrame = 0.0f;
steady_clock::duration total_time;

// random
random_device rd;
mt19937 gen(rd());
uniform_int_distribution<> dis_int; // for training position
uniform_real_distribution<> dis_real(0.0, 1.0); // for training texcoord

// input擷取框
glm::vec3 target_translate(100, 100, 150);
glm::vec3 target_scale(100, 100, 100);

// input體素選取矩陣
glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), target_scale);
glm::mat4 transMat  = glm::translate(glm::mat4(1.0f), target_translate);
glm::mat4 cropMatrix = transMat * scaleMat;

vector<glm::vec3> input_vector;
vector<Vertex_c> lattice_vertices;
vector<Vertex_c> color_model_vertices;
vector<Vertex_c> carved_vertices;
vector<Vertex_c> base_vertices; // for 鏤空的底層

DistanceField3D DF;
UIManager UI;
SOM_c SOM;
vector<LayerVoxels> layers_voxels;
double carving_threshold = 0.7;
char screenshot_name[128] = "../../screenshot/test.png";

void updateCropMatrix() {
    scaleMat = glm::scale(glm::mat4(1.0f), target_scale);
    transMat  = glm::translate(glm::mat4(1.0f), target_translate);
    cropMatrix = transMat * scaleMat;
}

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
    Shader_c color_shader("shader/color_shader.vs", "shader/color_shader.fs");
    Shader_c light_shader("shader/light_shader.vs", "shader/light_shader.fs");
    Shader_c tex_shader("shader/texture_shader.vs", "shader/texture_shader.fs");

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

    Object_c object_input_voxel;
    object_input_voxel.CreateObject(vertices[5], {});
    Object_c object_df_voxel;
    object_df_voxel.CreateObject(vertices[5], {});

    VolumeData df_data = ReadVolumeFile("../../raw/df_data/vase_dist.inf", "../../raw/df_data/vase_df.raw", false); // 學姊的
    VolumeData carved_df_data;
    // 取得體素表面頂點
    vector<Vertex_c> voxel_vertex = getModelVertices(df_data, 0);
    Object_c object_model;
    object_model.CreateObject(voxel_vertex, {});

    Object_c object_lattice;
    object_lattice.CreateObject(lattice_vertices, {}); // 初始化一個空的
    Object_c object_color_model;
    object_color_model.CreateObject(color_model_vertices, {}); 
    Object_c object_carved_layer;
    object_carved_layer.CreateObject(carved_vertices, {});
    Object_c object_base_layer;
    object_base_layer.CreateObject(base_vertices, {});
    
    
    
    SomData som_data(32, 32);
    som_data.setType(SQUARE);

    UI.init();

    unsigned int texture = 0;
    char image_path[256] = "../../picture/vase_pattern34.png";
    vector<vector<glm::vec3>> image = getTexture(image_path, texture);
    tex_shader.use();
    tex_shader.setInt("image", 0);
    
    // 顯示時統一縮放到一定範圍
    // 將(解析度 * voxel size)最高的那側作為分母縮放到512大小 並乘以voxel size
    double max_range = max(df_data.resolution[0] * df_data.voxel_size[0], max(df_data.resolution[1] * df_data.voxel_size[1], df_data.resolution[2] * df_data.voxel_size[2]));
    glm::vec3 ratio(512 / max_range); 
    ratio *= df_data.voxel_size;

    int distance_range = 0;
    
    bool is_layer_changed = false, is_screenshot = false;
    bool is_draw_input_BB = false, is_draw_input_voxels = true;
    bool is_draw_SOM = true, is_SOM_filled = true;
    bool is_draw_df_voxels = false, is_draw_model_surface = false, is_draw_bounding_box = false;
    int lattice_type;
    bool is_SOM_reset = false, is_SOM_start = false;
    bool is_draw_color_model = false, is_draw_carved_model = true, is_draw_base_layer = true;

    const char *type_names[] = {"position", "texture"};
    const char *filter_names[] = {"Linear", "Nearest"};
    int carved_layer = 1;

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

        // ============================== Imgui ==============================
        ImGui::StyleColorsLight();
        ImGui::NewFrame();
        UI.render(lightPos, camera.Position);
        
        ImGui::Begin("GUI");
        if(ImGui::InputInt("model layer (0 ~ x)", &carved_layer)) {
            if(carved_layer < 1)
                carved_layer = 1;
            else is_layer_changed = true;
        } 
        if(is_layer_changed) {
            if(ImGui::Button("Modify layer voxel")) {
                // 取得layer voxel
                glm::mat4 invCropMatrix = glm::inverse(cropMatrix);
                layers_voxels = getLayerVoxel(df_data, invCropMatrix, carved_layer);
                is_layer_changed = false;
            }
        }
        // input擷取框資訊
        ImGui::InputFloat3("input BB translate vector", &target_translate.x, "%.3f");
        ImGui::InputFloat3("input BB scaling vector", &target_scale.x, "%.3f");
        if(ImGui::Button("Get BB layer voxel")) {
            // 取得layer voxel
            glm::mat4 invCropMatrix = glm::inverse(cropMatrix);
            layers_voxels = getLayerVoxel(df_data, invCropMatrix, carved_layer);
            // 取得training input
            input_vector = layers_voxels[0].pos;
            object_input_voxel.UpdateInstances(input_vector);
            // 重製亂數範圍
            int size = input_vector.size();
            using param_t = std::uniform_int_distribution<int>::param_type; // 用左式取代右式一長串
            dis_int.param(param_t(0, size - 1));
            
            is_SOM_reset = true;
        }
        ImGui::Checkbox("input BB", &is_draw_input_BB); ImGui::SameLine();
        ImGui::Checkbox("input voxels", &is_draw_input_voxels); 
        ImGui::NewLine();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.0f, 0.5f, 0.5f, 1.0f)); 
        if(ImGui::Button("Screenshot")) 
            is_screenshot = true;
        ImGui::PopStyleColor(1);
        ImGui::SameLine();
        ImGui::InputText("Screenshot name", screenshot_name, sizeof(screenshot_name));
        // 模型不同距離值形成的集和
        ImGui::SliderInt("showing distance", &distance_range, ((int)df_data.minimum - 1), (int)df_data.maximum + 1);
        if(ImGui::Button("Update voxels")) {
            vector<glm::vec3> cube_pos;
            for(int i = 0; i < df_data.resolution[0]; i++)
                for(int j = 0; j < df_data.resolution[1]; j++)
                    for(int k = 0; k < df_data.resolution[2]; k++) 
                        if(df_data(i, j, k) > distance_range - 0.5 && df_data(i, j, k) < distance_range + 0.5)
                            cube_pos.push_back({i, j, k});
            cout<<"distance: "<<distance_range<<"  voxel number: "<<cube_pos.size()<<endl;
            object_df_voxel.UpdateInstances(cube_pos);
        }
        ImGui::Checkbox("model surface faces", &is_draw_model_surface); ImGui::SameLine();
        ImGui::Checkbox("model df-voxels", &is_draw_df_voxels); ImGui::SameLine();
        ImGui::Checkbox("model bounding box", &is_draw_bounding_box);
        ImGui::End();

        // SOM
        ImGui::Begin("SOM");
        if(ImGui::InputText("Image Path", image_path, sizeof(image_path), ImGuiInputTextFlags_EnterReturnsTrue)) 
            image = getTexture(image_path, texture);
        if(ImGui::Button("Load image"))
            image = getTexture(image_path, texture);
        // ImGui::Image((ImTextureID)(intptr_t)texture, ImVec2(256, 256)); // 可以在imgui顯示圖片
        if (ImGui::RadioButton("PLANE", &lattice_type, SQUARE)) { 
            som_data.setType(SQUARE);
            is_SOM_reset = true;
        } ImGui::SameLine();
        if (ImGui::RadioButton("CYLINDER", &lattice_type, CYLINDER)) { 
            som_data.setType(CYLINDER);
            is_SOM_reset = true;
        } ImGui::SameLine();
        if (ImGui::RadioButton("DONUT", &lattice_type, DONUT)) { 
            som_data.setType(DONUT);
            is_SOM_reset = true;
        } ImGui::SameLine();
        if (ImGui::RadioButton("CUBE", &lattice_type, CUBE)) { 
            som_data.setType(CUBE);
            is_SOM_reset = true;
        }
        if(ImGui::SliderInt("Training type", &SOM.training_type, 0, 1, type_names[SOM.training_type])) {
            SOM.initiateParameters(som_data);
            is_SOM_start = false;
        }
        if(SOM.training_type) {
            ImGui::InputInt("anchor value (>=1)", &SOM.anchor_value, 100, 1000);
        }
        if (ImGui::Button("Run")) is_SOM_start = true; ImGui::SameLine();
        if (ImGui::Button("Stop")) is_SOM_start = false;

        ImGui::Text("iter: %d", SOM.iter_now);
        ImGui::Text("radius: %f", SOM.radius_now);
        ImGui::Text("learning rate: %f", SOM.learning_rate_now);
        ImGui::Checkbox("SOM mesh", &is_draw_SOM); ImGui::SameLine();
        ImGui::Checkbox("Fill", &is_SOM_filled); 
        ImGui::NewLine();
        ImGui::SliderInt("Texture filter", reinterpret_cast<int*>(&tex_filter), 0, 1, filter_names[tex_filter]);
        if(ImGui::Button("Two step mapping")) {
            for(int l = 0; l < carved_layer; l++) {
                cout<<"Mapping payer: "<<l<<endl;
                twoStepMapping(layers_voxels[l], som_data, image);
            }
            // 儲存模型表面顏色頂點
            color_model_vertices = getColorModelVertices(layers_voxels[0], df_data, {1.0, 0.5, 0.0});
            object_color_model.RenewObject(color_model_vertices);
            // 儲存底層頂點
            base_vertices = getModelVertices(df_data, carved_layer);
            object_base_layer.RenewObject(base_vertices);
            // 重製carved_df_data，並把 0~layer - 2距離值的體素都設為模型，避免後續錯誤
            carved_df_data = df_data;
            for(int k = 0; k < df_data.resolution[2]; k++) {
                for(int j = 0; j < df_data.resolution[1]; j++) {
                    for(int i = 0; i < df_data.resolution[0]; i++) {
                        if(df_data(i, j, k) > 0 && df_data(i, j, k) < carved_layer)
                            carved_df_data(i, j, k) = 0;
                    }
                }
            }   
        }
        if(ImGui::Button("Engraving")) {
            for(int i = 0; i < carved_layer; i++)
                engravedCarving(layers_voxels[i], carved_df_data);
            carved_vertices = getModelVertices(carved_df_data, 0);
            object_carved_layer.RenewObject(carved_vertices);
        } ImGui::SameLine();
        if(ImGui::Button("Embossing")) {
            for(int i = 0; i < carved_layer; i++)
                embossedCarving(layers_voxels[i], carved_df_data);
            carved_vertices = getModelVertices(carved_df_data, 0);
            object_carved_layer.RenewObject(carved_vertices);
        }
        if(ImGui::Button("Dilation")) {
            dilation(carved_df_data, 3);
            carved_vertices = getModelVertices(carved_df_data, 0);
            object_carved_layer.RenewObject(carved_vertices);
        } ImGui::SameLine();
        if(ImGui::Button("Erosion")) {
            erosion(carved_df_data, 3);
            carved_vertices = getModelVertices(carved_df_data, 0);
            object_carved_layer.RenewObject(carved_vertices);
        }
        if(ImGui::Button("Opening")) {
            erosion(carved_df_data, 3);
            dilation(carved_df_data, 3);
            carved_vertices = getModelVertices(carved_df_data, 0);
            object_carved_layer.RenewObject(carved_vertices);
        } ImGui::SameLine();
        if(ImGui::Button("Closing")) {
            dilation(carved_df_data, 3);
            erosion(carved_df_data, 3);
            carved_vertices = getModelVertices(carved_df_data, 0);
            object_carved_layer.RenewObject(carved_vertices);
        }
        if(ImGui::Button("Reset carving effect")) {
            carved_df_data = df_data;
            // 把 0~layer - 1距離值的體素都設為模型，避免後續錯誤
            for(int k = 0; k < df_data.resolution[2]; k++) {
                for(int j = 0; j < df_data.resolution[1]; j++) {
                    for(int i = 0; i < df_data.resolution[0]; i++) {
                        if(df_data(i, j, k) > 0 && df_data(i, j, k) < carved_layer)
                            carved_df_data(i, j, k) = 0;
                    }
                }
            }
            object_carved_layer.RenewObject({});
        }
        ImGui::Checkbox("Color model", &is_draw_color_model); 
        ImGui::Checkbox("carved model", &is_draw_carved_model); ImGui::SameLine();
        ImGui::Checkbox("base layer", &is_draw_base_layer); 
        ImGui::End();

        // ============================== Control operation ==============================
        if(is_SOM_reset) { // 選取目標區域跟改lattice type時要reset
            // 取得SOM neuron mesh三角片並初始化
            SOM.initiateParameters(som_data);
            SOM.initiateLattice(input_vector, som_data);
            lattice_vertices = getLatticesVertices(som_data);
            object_lattice.RenewObject(lattice_vertices);

            total_time = steady_clock::duration::zero();
            is_SOM_start = false;
            is_SOM_reset = false;
        }
        if(is_SOM_start) {
            if(SOM.iter_now < SOM.iter_max) {
                auto start = steady_clock::now();
                for(int i = 0; i < 100; i++) { // 每次迴圈都訓練100次
                    if(SOM.training_type == 0) { // 代表訓練position
                        glm::vec3 input_pos;
                        input_pos = input_vector[dis_int(gen)];
                        SOM.runSOM(som_data.pos, input_pos);
                    } 
                    else { // 訓練texture
                        glm::vec3 input_texcoord = {dis_real(gen), dis_real(gen), 0.0f};
                        SOM.runSOM(som_data.texcoord, input_texcoord);
                    }
                }
                auto end = steady_clock::now();
                total_time += end - start;
                // 訓練一部分後更新lattice頂點
                lattice_vertices = getLatticesVertices(som_data);
                object_lattice.RenewObject(lattice_vertices);
            }
            else {
                cout<<"SOM訓練時間: "<<duration<double>(total_time).count()<<" s\n"<<endl;
                total_time = steady_clock::duration::zero();
                is_SOM_start = false;
            }
        }

        // ============================== Rendering ==============================
        // create model matrix
        glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        // create view matrix
        glm::mat4 view = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        // create projection matrix
        glm::mat4 projection = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first

        // ---------- active shader for color ----------
        color_shader.use();

        view = camera.GetViewMatrix();
        projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 10000.0f);
        color_shader.setMat4("view", view);
        color_shader.setMat4("projection", projection);
        color_shader.setVec3("lightPos", lightPos);
        color_shader.setVec3("viewPos", camera.Position);
        color_shader.setFloat("opacity", 1.0f);

        // 畫雕刻後的模型
        if(is_draw_carved_model) {
            glBindVertexArray(object_carved_layer.VAO_);
            model = glm::mat4(1.0f);
            model = glm::scale(model, ratio);
            color_shader.setMat4("model", model);
            color_shader.setVec3("color", glm::vec3(1.0, 0.5, 0.0));
            glDrawArrays(GL_TRIANGLES, 0, object_carved_layer.size);
        }
        
        // 畫模型底層
        if(is_draw_base_layer) {
            glBindVertexArray(object_base_layer.VAO_);
            model = glm::mat4(1.0f);
            model = glm::scale(model, ratio);
            color_shader.setMat4("model", model);
            color_shader.setVec3("color", glm::vec3(1.0, 0.2, 0.8));
            glDrawArrays(GL_TRIANGLES, 0, object_carved_layer.size);
        }
        // 畫模型表面 (距離值為0)
        if(is_draw_model_surface) {
            glBindVertexArray(object_model.VAO_);
            model = glm::mat4(1.0f);
            model = glm::scale(model, ratio);
            color_shader.setMat4("model", model);
            color_shader.setVec3("color", glm::vec3(1.0, 0.5, 0.0));
            glDrawArrays(GL_TRIANGLES, 0, object_model.size);
        }

        // 畫不同距離的模型
        if(is_draw_df_voxels) {
            glBindVertexArray(object_df_voxel.VAO_);
            model = glm::mat4(1.0f);
            model = glm::scale(model, ratio);
            color_shader.setMat4("model", model);
            color_shader.setVec3("color", glm::vec3(1.0, 0.5, 0.0));
            glDrawArraysInstanced(GL_TRIANGLES, 0, object_df_voxel.size, object_df_voxel.instanceCount); 
        }

        // 畫選取框內的Input voxels
        if(is_draw_input_voxels) {
            glBindVertexArray(object_input_voxel.VAO_);
            model = glm::mat4(1.0f);
            model = glm::scale(model, ratio);
            color_shader.setMat4("model", model);
            color_shader.setVec3("color", glm::vec3(1.0, 0.5, 0.0));
            glDrawArraysInstanced(GL_TRIANGLES, 0, object_input_voxel.size, object_input_voxel.instanceCount); 
        }

        // ---------- active shader for texture ----------
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        tex_shader.use();
        tex_shader.setVec3("lightPos", lightPos);
        tex_shader.setVec3("viewPos", camera.Position);
        tex_shader.setMat4("view", view);
        tex_shader.setMat4("projection", projection);
        tex_shader.setFloat("opacity", 1.0f);
        
        // 畫SOM mesh
        if(is_draw_SOM) {
            glBindVertexArray(object_lattice.VAO_);
            model = glm::scale(glm::mat4(1.0f), ratio);
            tex_shader.setMat4("model", model);
            if(is_SOM_filled) {
                glDrawArrays(GL_TRIANGLES, 0, object_lattice.size);
            }
            else {
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                glDrawArrays(GL_TRIANGLES, 0, object_lattice.size);
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            }
        }

        // ---------- active shader for background ----------
        light_shader.use();
        light_shader.setVec3("lightPos", lightPos);
        light_shader.setVec3("viewPos", camera.Position);
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

        if(is_draw_color_model) {
            glBindVertexArray(object_color_model.VAO_);
            model = glm::mat4(1.0f);
            model = glm::scale(model, ratio);
            light_shader.setMat4("model", model);
            glDrawArrays(GL_TRIANGLES, 0, object_color_model.size);
        }
        
        // 畫 model BB
        if(is_draw_bounding_box) {
            glDepthMask(GL_FALSE); 
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

        // 畫選取方塊 ( input BB )
        if(is_draw_input_BB) {
            color_shader.use();
            glBindVertexArray(cube.VAO_);
            model = glm::scale(glm::mat4(1.0f), ratio);
            model = model * cropMatrix;
            color_shader.setMat4("model", model);
            color_shader.setVec3("color", glm::vec3(1.0, 0.9, 0.0));
            color_shader.setFloat("opacity", 0.6f);
            glDepthMask(GL_FALSE); 
            glDrawArrays(GL_TRIANGLES, 0, cube.size);
            glDepthMask(GL_TRUE); 
        }
        
        if(is_screenshot) {
            saveOpenGLScreenshot(screenshot_name);
            is_screenshot = false;
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
    if (ImGui::GetIO().WantCaptureKeyboard) // 避免在imgui視窗打字觸發鍵盤事件
        return;
    if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
        moveObject = (moveObject + 1) % 3;
    }
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (ImGui::GetIO().WantCaptureKeyboard) // 避免在imgui視窗打字觸發鍵盤事件
        return;
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if(moveObject == 0) // 視角
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
    else if(moveObject == 1) // light
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
    else if(moveObject == 2) // 擷取框範圍
    {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            target_translate.z -= 100.0f * deltaTime; 
            updateCropMatrix();
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            target_translate.z += 100.0f * deltaTime; 
            updateCropMatrix();
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            target_translate.x -= 100.0f * deltaTime; 
            updateCropMatrix();
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            target_translate.x += 100.0f * deltaTime; 
            updateCropMatrix();
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            target_translate.y += 100.0f * deltaTime; 
            updateCropMatrix();
        }
        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
            target_translate.y -= 100.0f * deltaTime; 
            updateCropMatrix();
        }

        if (glfwGetKey(window, GLFW_KEY_KP_4) == GLFW_PRESS) {
            target_scale.x -= 100.0f * deltaTime; 
            updateCropMatrix();
        }
        if (glfwGetKey(window, GLFW_KEY_KP_6) == GLFW_PRESS) {
            target_scale.x += 100.0f * deltaTime; 
            updateCropMatrix();
        }
        if (glfwGetKey(window, GLFW_KEY_KP_2) == GLFW_PRESS) {
            target_scale.y -= 100.0f * deltaTime; 
            updateCropMatrix();
        }
        if (glfwGetKey(window, GLFW_KEY_KP_8) == GLFW_PRESS) {
            target_scale.y += 100.0f * deltaTime; 
            updateCropMatrix();
        }
        if (glfwGetKey(window, GLFW_KEY_KP_0) == GLFW_PRESS) {
            target_scale.z -= 100.0f * deltaTime; 
            updateCropMatrix();
        }
        if (glfwGetKey(window, GLFW_KEY_KP_5) == GLFW_PRESS) {
            target_scale.z += 100.0f * deltaTime; 
            updateCropMatrix();
        }
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

// 找模型表面體素，取得繪製體素的表面
// 或許可增進功能?
vector<Vertex_c> getModelVertices(VolumeData &df_data, int distance_value) {
    vector<Vertex_c> vertex;
    auto start = steady_clock::now();
    glm::ivec3 resolution = df_data.resolution;

    // 建立一個 lambda 輔助函式，用來插入頂點並自動加上位移
    auto insertAndOffsetFace = [&](const vector<Vertex_c>& face_vertices, glm::ivec3 offset) {
        size_t old_size = vertex.size();
        vertex.insert(vertex.end(), face_vertices.begin(), face_vertices.end());
        size_t new_size = vertex.size();
        
        for(size_t v = old_size; v < new_size; v++) {
            vertex[v].position.x += offset.x;
            vertex[v].position.y += offset.y;
            vertex[v].position.z += offset.z;
        }
    };

    for(int k = 0; k < resolution[2]; k++) {
        for(int j = 0; j < resolution[1]; j++) {
            for(int i = 0; i < resolution[0]; i++) {
                if(df_data(i, j, k) == distance_value) { // 遇到model檢查鄰居是否是空氣
                    if((i == 0) || (df_data(i - 1, j, k)) != distance_value) { // -x
                        insertAndOffsetFace(cube_face_vertices[0], {i, j, k});
                    }
                    if((i == df_data.resolution[0] - 1) || (df_data(i + 1, j, k)) != distance_value) { // +x
                        insertAndOffsetFace(cube_face_vertices[1], {i, j, k});
                    }
                    if((j == 0) || (df_data(i, j - 1, k)) != distance_value) { // -y
                        insertAndOffsetFace(cube_face_vertices[2], {i, j, k});
                    }
                    if((j == df_data.resolution[1] - 1) || (df_data(i, j + 1, k)) != distance_value) { // +y
                        insertAndOffsetFace(cube_face_vertices[3], {i, j, k});
                    }
                    if((k == 0) || (df_data(i, j, k - 1)) != distance_value) { // -z
                        insertAndOffsetFace(cube_face_vertices[4], {i, j, k});
                    }
                    if((k == df_data.resolution[2] - 1) || (df_data(i, j, k + 1)) != distance_value) { // +z
                        insertAndOffsetFace(cube_face_vertices[5], {i, j, k});
                    }
                }
            }
        }
    }
    auto end = steady_clock::now();
    auto diff = duration<double>(end - start);
    cout<<"計算模型表面時間: "<<diff.count()<<" s\n"<<endl;

    return vertex;
}

// 找在範圍內的模型每層距離值的體素，其中距離值為0即代表SOM訓練input
vector<LayerVoxels> getLayerVoxel(VolumeData &df_data, glm::mat4 matrix, int layer) {
    vector<LayerVoxels> layers_voxels(layer - 1); // 去掉底層
    auto start = steady_clock::now();
    glm::ivec3 resolution = df_data.resolution;
    
    for(int k = 0; k < resolution[2]; k++) {
        for(int j = 0; j < resolution[1]; j++) {
            for(int i = 0; i < resolution[0]; i++) {
                glm::vec4 voxel(i + 0.5f, j + 0.5f, k + 0.5f, 1.0f); // model space
                glm::vec4 local_space = matrix * voxel; // transform to LCS
                glm::vec3 local_pos = glm::vec3(local_space) / local_space.w; // get local position
                
                // 直接檢查是否在LCS的 0~1 內
                if (local_pos.x >= 0.0f && local_pos.x <= 1.0f && local_pos.y >= 0.0f && local_pos.y <= 1.0f && local_pos.z >= 0.0f && local_pos.z <= 1.0f) {
                    for(int l = 0; l < layer - 1; l++) {
                        if(df_data(i, j, k) == l)
                            layers_voxels[l].pos.push_back({i, j, k});
                    }
                }
            }
        }
    }
    for(int i = 0; i < layer - 1; i++)
        cout<<"layer"<<" "<<i<<": "<<layers_voxels[i].pos.size()<<endl;
        
    auto end = steady_clock::now();
    auto diff = duration<double>(end - start);
    cout<<"擷取所有層之體素時間: "<<diff.count()<<" s\n"<<endl;
    return layers_voxels;
}

// 得到lattice的mesh頂點資訊
vector<Vertex_c> getLatticesVertices(SomData som_data) {
    vector<Vertex_c> vertices;
    int lattice_type = som_data.lattice_type;
    int *resolution = som_data.resolution;

    for(int k = 0; k < resolution[2]; k++)
       for(int j = 0; j < resolution[1] - 1; j++)
            for(int i = 0; i < resolution[0] - 1; i++) {
                // vertices position
                glm::vec3 v0(som_data(i, j, k));
                glm::vec3 v1(som_data(i + 1, j, k));
                glm::vec3 v2(som_data(i + 1, j + 1, k));
                glm::vec3 v3(som_data(i, j + 1, k));
                // texcoord
                glm::vec3 t0(som_data(i, j));
                glm::vec3 t1(som_data(i + 1, j));
                glm::vec3 t2(som_data(i + 1, j + 1));
                glm::vec3 t3(som_data(i, j + 1));
                // normal
                glm::vec3 n0 = glm::normalize(glm::cross(v1 - v0, v3 - v0)); // 左下三角片
                glm::vec3 n1 = glm::normalize(glm::cross(v3 - v2, v1 - v2)); // 右上三角片

                vertices.push_back({{v0.x, v0.y, v0.z}, {}, {t0.x, t0.y, t0.z}, {n0.x, n0.y, n0.z}});
                vertices.push_back({{v1.x, v1.y, v1.z}, {}, {t1.x, t1.y, t1.z}, {n0.x, n0.y, n0.z}});
                vertices.push_back({{v3.x, v3.y, v3.z}, {}, {t3.x, t3.y, t3.z}, {n0.x, n0.y, n0.z}});

                vertices.push_back({{v1.x, v1.y, v1.z}, {}, {t1.x, t1.y, t1.z}, {n1.x, n1.y, n1.z}});
                vertices.push_back({{v2.x, v2.y, v2.z}, {}, {t2.x, t2.y, t2.z}, {n1.x, n1.y, n1.z}});
                vertices.push_back({{v3.x, v3.y, v3.z}, {}, {t3.x, t3.y, t3.z}, {n1.x, n1.y, n1.z}});
            }

    return vertices;
}

vector<vector<glm::vec3>> getTexture(const char *filename, unsigned int &texture) {
    if(texture == 0) // 第一次需要初始化
        glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char *data = stbi_load(filename, &width, &height, &nrChannels, 3);
    if (data) {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    } else {
        cout<<"load image failed"<<endl;
        return {};
    }

    vector<vector<glm::vec3>> image(height, vector<glm::vec3>(width));

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            // 計算該像素在一維陣列中的起始位置
            int index = (j * width + i) * 3;

            int r = data[index + 0];
            int g = data[index + 1];
            int b = data[index + 2];

            image[j][i] = glm::vec3(r, g, b) / 255.0f;
        }
    }
    stbi_image_free(data);
    cout<<"Image: "<<width<<" x "<<height<<endl<<endl;;
    return image;
}

// 將mapping後的顏色值存成頂點顏色，不再mapping範圍的另外給值，只存模型表面 (距離值0)
vector<Vertex_c> getColorModelVertices(LayerVoxels &layer_voxels, VolumeData &df_data, glm::vec3 uncarved_color) {
    vector<Vertex_c> vertices;
    auto start = steady_clock::now();
    
    glm::ivec3 resolution = df_data.resolution;
    int size = layer_voxels.pos.size();
    VolumeData color_data = df_data; // 複製一份來存顏色
    color_data.voxel_data.assign(color_data.voxel_data.size(), -1); // 顏色先清-1
    for(int l = 0; l < size; l++) {
        // 取得index
        int i = layer_voxels.pos[l].x;
        int j = layer_voxels.pos[l].y;
        int k = layer_voxels.pos[l].z;
        // 轉成灰階顏色
        double gray_color = 0.299 * layer_voxels.color[l].r + 0.587 * layer_voxels.color[l].g + 0.114 * layer_voxels.color[l].b;
        // 存進color data
        color_data(i, j, k) = gray_color;
    }

    // 建立一個 lambda 輔助函式，用來插入頂點並自動加上位移並補上對應顏色
    auto insertAndOffsetFace = [&](const vector<Vertex_c>& face_vertices, glm::ivec3 offset, double color) {
        size_t old_size = vertices.size();
        vertices.insert(vertices.end(), face_vertices.begin(), face_vertices.end());
        size_t new_size = vertices.size();
        float f_color = static_cast<float>(color);
        for(size_t v = old_size; v < new_size; v++) {
            vertices[v].position.x += offset.x;
            vertices[v].position.y += offset.y;
            vertices[v].position.z += offset.z;
            if(color == -1) { // 不在擷取範圍就填上自定義顏色
                vertices[v].color = {uncarved_color.r, uncarved_color.g, uncarved_color.b};
            } else {
                vertices[v].color = {f_color, f_color, f_color};
            }
        }
    };

    for(int k = 0; k < resolution[2]; k++) {
        for(int j = 0; j < resolution[1]; j++) {
            for(int i = 0; i < resolution[0]; i++) {
                if(df_data(i, j, k) == 0) { // 遇到model檢查鄰居是否是空氣
                    if((i == 0) || (df_data(i - 1, j, k)) != 0) { // -x
                        insertAndOffsetFace(cube_face_vertices[0], {i, j, k}, color_data(i, j, k));
                    }
                    if((i == df_data.resolution[0] - 1) || (df_data(i + 1, j, k)) != 0) { // +x
                        insertAndOffsetFace(cube_face_vertices[1], {i, j, k}, color_data(i, j, k));
                    }
                    if((j == 0) || (df_data(i, j - 1, k)) != 0) { // -y
                        insertAndOffsetFace(cube_face_vertices[2], {i, j, k}, color_data(i, j, k));
                    }
                    if((j == df_data.resolution[1] - 1) || (df_data(i, j + 1, k)) != 0) { // +y
                        insertAndOffsetFace(cube_face_vertices[3], {i, j, k}, color_data(i, j, k));
                    }
                    if((k == 0) || (df_data(i, j, k - 1)) != 0) { // -z
                        insertAndOffsetFace(cube_face_vertices[4], {i, j, k}, color_data(i, j, k));
                    }
                    if((k == df_data.resolution[2] - 1) || (df_data(i, j, k + 1)) != 0) { // +z
                        insertAndOffsetFace(cube_face_vertices[5], {i, j, k}, color_data(i, j, k));
                    }
                }
            }
        }
    }
    auto end = steady_clock::now();
    auto diff = duration<double>(end - start);
    cout<<"計算顏色模型表面時間: "<<diff.count()<<" s\n"<<endl;

    return vertices;
}

// 將df_data的某一層做陰刻(圖案挖掉)
void engravedCarving(LayerVoxels &layer_voxels, VolumeData &carved_df_data) {
    int size = layer_voxels.pos.size();
    for(int l = 0; l < size; l++) {
        // 取得index
        int i = layer_voxels.pos[l].x;
        int j = layer_voxels.pos[l].y;
        int k = layer_voxels.pos[l].z;
        // 轉成灰階顏色
        double gray_color = 0.299 * layer_voxels.color[l].r + 0.587 * layer_voxels.color[l].g + 0.114 * layer_voxels.color[l].b;
        if(gray_color < carving_threshold)
            carved_df_data(i, j, k) = 1;
    }
}

// 將df_data的某一層做陽刻(圖案以外挖掉)
void embossedCarving(LayerVoxels &layer_voxels, VolumeData &carved_df_data) {
    int size = layer_voxels.pos.size();
    for(int l = 0; l < size; l++) {
        // 取得index
        int i = layer_voxels.pos[l].x;
        int j = layer_voxels.pos[l].y;
        int k = layer_voxels.pos[l].z;
        // 轉成灰階顏色
        double gray_color = 0.299 * layer_voxels.color[l].r + 0.587 * layer_voxels.color[l].g + 0.114 * layer_voxels.color[l].b;
        if(gray_color > carving_threshold)
            carved_df_data(i, j, k) = 1;
    }
}

// 縮減黑色範圍，白色範圍增加(空氣)(距離值!=0) // 這個program的使用跟一般運算定義相反(黑白對象互換)
void erosion(VolumeData &carved_df_data, int threshold) {
    glm::ivec3 resolution = carved_df_data.resolution;
    glm::ivec3 offset[6] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};

    VolumeData temp_data = carved_df_data;
    for(int k = 1; k < resolution[2] - 1; k++) { // 偷懶跳過邊緣，但體素化時正常已經有padding過了應該沒差
        for(int j = 1; j < resolution[1] - 1; j++) {
            for(int i = 1; i < resolution[0] - 1; i++) {
                // 六個鄰居有threshld個以上白色就設為白色(空氣)(距離值!=0)
                if(temp_data(i, j, k) == 0) { // 利用只有模型需要判斷來加速，之後距離值改變可能有危險!
                    int count = 0;
                    for(int d = 0; d < 6; d++) {
                        glm::ivec3 index = glm::ivec3(i, j, k) + offset[d]; 
                        if(temp_data(index.x, index.y, index.z) > 0)
                            count++;
                    }
                    if(count >= threshold)
                        carved_df_data(i, j, k) = 1;
                }
            }
        }
    }
}

// 擴增黑色範圍(模型)(距離值=0)
void dilation(VolumeData &carved_df_data, int threshold) {
    glm::ivec3 resolution = carved_df_data.resolution;
    glm::ivec3 offset[6] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};

    VolumeData temp_data = carved_df_data;
    for(int k = 1; k < resolution[2] - 1; k++) { // 偷懶跳過邊緣，但體素化時正常已經有padding過了應該沒差
        for(int j = 1; j < resolution[1] - 1; j++) {
            for(int i = 1; i < resolution[0] - 1; i++) {
                // 六個鄰居有threshld個以上黑色就設為黑色(模型)(距離值=0)
                if(temp_data(i, j, k) == 1) { // 利用只有模型周圍體素需要判斷來加速，之後距離值改變可能有危險!
                    int count = 0;
                    for(int d = 0; d < 6; d++) {
                        glm::ivec3 index = glm::ivec3(i, j, k) + offset[d]; 
                        if(temp_data(index.x, index.y, index.z) == 0)
                            count++;
                    }
                    if(count >= threshold)
                        carved_df_data(i, j, k) = 0;
                }
            }
        }
    }
}
void saveOpenGLScreenshot(const string& filename) {
    // 1. 取得目前 Viewport 尺寸 (x, y, width, height)
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    int width  = viewport[2];
    int height = viewport[3];

    if (width <= 0 || height <= 0) {
        cerr << "[Screenshot] Invalid viewport size.\n";
        return;
    }

    // 2. 確保讀取像素時列對齊為 1 byte，避免寬度非 4 的倍數時畫面錯位
    GLint prevAlignment = 4;
    glGetIntegerv(GL_PACK_ALIGNMENT, &prevAlignment);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    // 3. 讀取目前讀取緩衝區 (預設為 GL_BACK / 目前綁定的 FBO) 的 RGB 像素
    vector<unsigned char> pixels(static_cast<size_t>(width) * height * 3);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // 還原原本的對齊設定
    glPixelStorei(GL_PACK_ALIGNMENT, prevAlignment);

    // 4. 垂直翻轉影像 (OpenGL 底部為 y=0，PNG 頂部為 y=0)
    stbi_flip_vertically_on_write(1);

    // 5. 寫出 PNG (stride = width * 3 bytes)
    int success = stbi_write_png(filename.c_str(), width, height, 3, pixels.data(), width * 3);
    if (!success) {
        cerr << "[Screenshot] Failed to write PNG: " << filename << "\n";
    }
}