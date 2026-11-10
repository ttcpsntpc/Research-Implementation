#version 330 core
out vec4 FragColor;

in vec3 ourColor;
in vec3 TexCoord;
in vec3 Normal;
in vec3 FragPos;

uniform float opacity;

void main(){
    FragColor = vec4(ourColor, opacity);
}
