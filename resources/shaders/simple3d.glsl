#version 460

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;

#ifdef VERTEX_SHADER

in vec3 in_position;
in vec3 in_normal;

out vec3 N;
out vec3 pos;

void main(){
    gl_Position = proj*view*model*vec4(in_position, 1.0);
    N = mat3(transpose(inverse(model))) * in_normal;
    N = normalize(N);
}

#elif FRAGMENT_SHADER

in vec3 N;
in vec3 pos;
layout(location = 0) out vec4 frag;

void main(){
    vec3 lightColor = vec3(1.0);
    vec3 lightDir = normalize(vec3(0, 0, -1));
    vec3 objColor = N;

    float diff = max(dot(N, -lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 result = (vec3(0.1) + diffuse) * objColor;
    frag = vec4(result, 1.0);
    // frag = vec4(N, 1.0);
}

#endif
