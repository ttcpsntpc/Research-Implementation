#version 330 core
out vec4 FragColor;

in vec3 ourColor;
in vec3 TexCoord;
in vec3 Normal;
in vec3 FragPos;

uniform sampler2D image;
uniform float opacity;

void main(){
    vec3 P = FragPos;
    FragColor = vec4(texture(image, TexCoord.st).rgb, opacity);
}
