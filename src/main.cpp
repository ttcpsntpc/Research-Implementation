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
void getModelSurface(VolumeData &df_data, vector<Vertex_c> &vertex);
vector<glm::vec3> getTrainingVoxel(VolumeData &df_data, glm::mat4 matrix);
vector<Vertex_c> getLatticesVertices(SomData som_data);

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

DistanceField3D DF;
UIManager UI;
SOM_c SOM;

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

    // 取得體素表面頂點
    vector<Vertex_c> voxel_vertex;
    getModelSurface(df_data, voxel_vertex);
    Object_c object_model;
    object_model.CreateObject(voxel_vertex, {});

    Object_c object_lattice;
    object_lattice.CreateObject(lattice_vertices, {}); // 初始化一個空的
    
    SomData som_data(32, 32);
    som_data.setType(SQUARE);

    UI.init();

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    int width, height, nrChannels;
    unsigned char *data = stbi_load("../../picture/chest.png", &width, &height, &nrChannels, 0);
    if (data) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    } else cout<<"load image failed"<<endl;
    tex_shader.use();
    tex_shader.setInt("image", 0);
    
    // 顯示時統一縮放到一定範圍
    // 將(解析度 * voxel size)最高的那側作為分母縮放到512大小 並乘以voxel size
    double max_range = max(df_data.resolution[0] * df_data.voxel_size[0], max(df_data.resolution[1] * df_data.voxel_size[1], df_data.resolution[2] * df_data.voxel_size[2]));
    glm::vec3 ratio(512 / max_range); 
    ratio *= df_data.voxel_size;

    int distance_range = 0;
    
    bool is_draw_input_BB = true, is_draw_input_voxels = true;
    bool is_draw_SOM = true, is_SOM_filled = false;
    bool is_draw_df_voxels = false, is_draw_model_surface = false, is_draw_bounding_box = false;

    int lattice_type = 0, training_type = 0;
    LatticesType TEMP = som_data.lattice_type;
    bool is_SOM_reset = false, is_SOM_start = false;
    const char *elems_names[] = {"position", "texture"};

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
        ImGui::NewFrame();
        UI.render(lightPos, camera.Position);
        
        ImGui::Begin("GUI");
        // input擷取框資訊
        ImGui::InputFloat3("input BB translate vector", &target_translate.x, "%.3f");
        ImGui::InputFloat3("input BB scaling vector", &target_scale.x, "%.3f");
        if(ImGui::Button("get input voxel")) {
            // 取得training voxel
            glm::mat4 invCropMatrix = glm::inverse(cropMatrix);
            input_vector = getTrainingVoxel(df_data, invCropMatrix);
            object_input_voxel.UpdateInstances(input_vector);
            // 重製亂數範圍
            int size = input_vector.size();
            using param_t = std::uniform_int_distribution<int>::param_type; // 用左式取代右式一長串
            dis_int.param(param_t(0, size - 1));
            
            is_SOM_reset = true;
        }
        ImGui::Checkbox("input BB", &is_draw_input_BB); ImGui::SameLine();
        ImGui::Checkbox("input voxels", &is_draw_input_voxels); 

        
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
        ImGui::Checkbox("model surface faces", &is_draw_model_surface); 
        ImGui::Checkbox("model df-voxels", &is_draw_df_voxels);
        ImGui::Checkbox("model bounding box", &is_draw_bounding_box);
        ImGui::End();

        // SOM
        ImGui::Begin("SOM");
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
        if(ImGui::SliderInt("Training type", &training_type, 0, 1, elems_names[training_type])) {
            if(training_type == 1 && som_data.lattice_type == CUBE) { // texture // 為了type是CUBE時仍以平面做訓練
                TEMP = som_data.lattice_type; som_data.setType(SQUARE);
                SOM.initiateParameters(som_data); som_data.setType(TEMP);
            } else
                SOM.initiateParameters(som_data);
            is_SOM_start = false;
        }
        if (ImGui::Button("Run")) is_SOM_start = true; ImGui::SameLine();
        if (ImGui::Button("Stop")) is_SOM_start = false;

        ImGui::Text("iter: %d", SOM.iter_now);
        ImGui::Text("radius: %f", SOM.radius_now);
        ImGui::Text("learning rate: %f", SOM.learning_rate_now);
        ImGui::Checkbox("SOM mesh", &is_draw_SOM); 
        ImGui::Checkbox("Fill", &is_SOM_filled); 
        ImGui::End();

        if(is_SOM_reset) { // 選取目標區域跟改lattice type跟時要reset
            // 取得SOM neuron mesh三角片並初始化
            SOM.initiateParameters(som_data);
            SOM.initiateLattice(input_vector, som_data);
            lattice_vertices = getLatticesVertices(som_data);
            object_lattice.RenewObject(lattice_vertices);

            if(training_type == 1) { // texture // 就為了type是CUBE時仍以平面做訓練
                TEMP = som_data.lattice_type; som_data.setType(SQUARE);
                SOM.initiateParameters(som_data); som_data.setType(TEMP);
            }

            is_SOM_start = false;
            is_SOM_reset = false;
        }
        if(is_SOM_start) {
            if(SOM.iter_now < SOM.iter_max) {
                for(int i = 0; i < 100; i++) { // 每次迴圈都訓練100次
                    if(training_type == 0) { // 代表訓練position
                        glm::vec3 input_pos;
                        input_pos = input_vector[dis_int(gen)];
                        SOM.runSOM(som_data.pos, input_pos);
                    } 
                    else { // 訓練texture
                        glm::vec3 input_texcoord = {dis_real(gen), dis_real(gen), 0.0f};
                        SOM.runSOM(som_data.texcoord, input_texcoord);
                    }
                }
                // 訓練一部分後更新lattice頂點
                lattice_vertices = getLatticesVertices(som_data);
                object_lattice.RenewObject(lattice_vertices);
            }
            else is_SOM_start = false;
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

        // 畫模型表面
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

        // 畫選取方塊 ( input BB )
        if(is_draw_input_BB) {
            glBindVertexArray(cube.VAO_);
            model = glm::scale(glm::mat4(1.0f), ratio);
            model = model * cropMatrix;
            color_shader.setMat4("model", model);
            color_shader.setVec3("color", glm::vec3(1.0, 0.9, 0.0));
            color_shader.setFloat("opacity", 0.6f);
            glDepthMask(GL_FALSE); 
            glDrawArrays(GL_TRIANGLES, 0, cube.size);
            glDepthMask(GL_TRUE); 
            color_shader.setFloat("opacity", 1.0f);
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
        moveObject = (moveObject + 1) % 3;
    }
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
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
void getModelSurface(VolumeData &df_data, vector<Vertex_c> &vertex) {
    
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

    for(int i = 0; i < df_data.resolution[0]; i++) {
        for(int j = 0; j < df_data.resolution[1]; j++) {
            for(int k = 0; k < df_data.resolution[2]; k++) {
                if(df_data(i, j, k) == 0) { // 遇到model檢查鄰居是否是空氣
                    if((i == 0) || (df_data(i - 1, j, k)) != 0) { // -x
                        insertAndOffsetFace(cube_face_vertices[0], {i, j, k});
                    }
                    if((i == df_data.resolution[0] - 1) || (df_data(i + 1, j, k)) != 0) { // +x
                        insertAndOffsetFace(cube_face_vertices[1], {i, j, k});
                    }
                    if((j == 0) || (df_data(i, j - 1, k)) != 0) { // -y
                        insertAndOffsetFace(cube_face_vertices[2], {i, j, k});
                    }
                    if((j == df_data.resolution[1] - 1) || (df_data(i, j + 1, k)) != 0) { // +y
                        insertAndOffsetFace(cube_face_vertices[3], {i, j, k});
                    }
                    if((k == 0) || (df_data(i, j, k - 1)) != 0) { // -z
                        insertAndOffsetFace(cube_face_vertices[4], {i, j, k});
                    }
                    if((k == df_data.resolution[2] - 1) || (df_data(i, j, k + 1)) != 0) { // +z
                        insertAndOffsetFace(cube_face_vertices[5], {i, j, k});
                    }
                }
            }
        }
    }
}

// 找在範圍內的模型表面體素，當作SOM input
vector<glm::vec3> getTrainingVoxel(VolumeData &df_data, glm::mat4 matrix) {
    vector<glm::vec3> input_vector;
    
    for(int i = 0; i < df_data.resolution[0]; i++) {
        for(int j = 0; j < df_data.resolution[1]; j++) {
            for(int k = 0; k < df_data.resolution[2]; k++) {
                if(df_data(i, j, k) == 0) { // 可改進，讓真正在表面的體素才納入
                    glm::vec4 voxel(i + 0.5f, j + 0.5f, k + 0.5f, 1.0f); // model space
                    glm::vec4 local_space = matrix * voxel; // transform to LCS
                    glm::vec3 local_pos = glm::vec3(local_space) / local_space.w; // get local position
                    
                    // 直接檢查是否在 0~1 內
                    if (local_pos.x >= 0.0f && local_pos.x <= 1.0f &&
                        local_pos.y >= 0.0f && local_pos.y <= 1.0f &&
                        local_pos.z >= 0.0f && local_pos.z <= 1.0f)  
                        input_vector.push_back(glm::vec3(i, j, k));
                }
            }
        }
    }
    return input_vector;
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