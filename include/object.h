#ifndef OBJECT_H
#define OBJECT_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "Vertex.h"

class Object_c
{
public:
    unsigned int VAO_ = 0;
    unsigned int VBO_ = 0;
    unsigned int EBO_ = 0;
    unsigned int instance_VBO_ = 0;

    unsigned int size = 0;          // 頂點的數量
    unsigned int instanceCount = 1; // Instance 的數量 (預設一個)

    // 追蹤 GPU 記憶體容量
    size_t vboCapacityBytes_ = 0;
    size_t instanceCapacityBytes_ = sizeof(glm::vec3); // 預設一個vec3的大小
    
    Object_c()
    {
    }

    ~Object_c()
    {
        if (VAO_ != 0) glDeleteVertexArrays(1, &VAO_);
        if (VBO_ != 0) glDeleteBuffers(1, &VBO_);
        if (EBO_ != 0) glDeleteBuffers(1, &EBO_);
        if (instance_VBO_ != 0) glDeleteBuffers(1, &instance_VBO_);
    }

    void CreateObject(const std::vector<Vertex_c> &vertices, const std::vector<unsigned int> &indices)
    {
        glGenVertexArrays(1, &VAO_);
        glGenBuffers(1, &VBO_);
        glGenBuffers(1, &instance_VBO_);

        glBindVertexArray(VAO_);
        glBindBuffer(GL_ARRAY_BUFFER, VBO_);

        vboCapacityBytes_ = vertices.size() * sizeof(Vertex_c);
        glBufferData(GL_ARRAY_BUFFER, vboCapacityBytes_, vertices.data(), GL_DYNAMIC_DRAW);

        size = vertices.size();

        if (!indices.empty())
        {
            glGenBuffers(1, &EBO_);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
        }

        // attributes
        // position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex_c), 0);
        glEnableVertexAttribArray(0);

        // color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex_c), (void *)sizeof(Vertex_c::Position));
        glEnableVertexAttribArray(1);

        // texture attribute
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex_c), (void *)(sizeof(Vertex_c::Position) + sizeof(Vertex_c::Color)));
        glEnableVertexAttribArray(2);

        // normal attribute
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex_c), (void *)(sizeof(Vertex_c::Position) + sizeof(Vertex_c::Color) + sizeof(Vertex_c::Texture)));
        glEnableVertexAttribArray(3);
        
        glm::vec3 zero_vector(0.0f);
        glBindBuffer(GL_ARRAY_BUFFER, instance_VBO_);
        glBufferData(GL_ARRAY_BUFFER, instanceCapacityBytes_, &zero_vector, GL_DYNAMIC_DRAW);

        // instance for position
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glEnableVertexAttribArray(4);
        glVertexAttribDivisor(4, 1); 
    }

    void RenewObject(const std::vector<Vertex_c> &vertices)
    {
        glBindBuffer(GL_ARRAY_BUFFER, VBO_);

        size_t newSizeBytes = vertices.size() * sizeof(Vertex_c);
        if (newSizeBytes > vboCapacityBytes_) {
            vboCapacityBytes_ = std::max(newSizeBytes, vboCapacityBytes_ + vboCapacityBytes_ / 2);
            glBufferData(GL_ARRAY_BUFFER, vboCapacityBytes_, vertices.data(), GL_DYNAMIC_DRAW);
        } else {
            glBufferSubData(GL_ARRAY_BUFFER, 0, newSizeBytes, vertices.data());
        }

        size = vertices.size();
    }

    void UpdateInstances(const std::vector<glm::vec3> &pos) 
    {
        glBindBuffer(GL_ARRAY_BUFFER, instance_VBO_);
        size_t newSizeBytes = pos.size() * sizeof(glm::vec3);

        // 容量管理邏輯
        if (newSizeBytes > instanceCapacityBytes_) 
        {
            // 擴容 1.5 倍，並使用 GL_DYNAMIC_DRAW
            instanceCapacityBytes_ = std::max(newSizeBytes, instanceCapacityBytes_ + instanceCapacityBytes_ / 2);
            glBufferData(GL_ARRAY_BUFFER, instanceCapacityBytes_, pos.data(), GL_DYNAMIC_DRAW);
        } 
        else 
        {
            // 容量足夠，直接覆寫
            glBufferSubData(GL_ARRAY_BUFFER, 0, newSizeBytes, pos.data());
        }

        instanceCount = pos.size();
    }
};

#endif