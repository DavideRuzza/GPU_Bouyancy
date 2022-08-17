#version 460

// layout(binding=0) uniform sampler2D testure;
// uniform vec3 color;

#ifdef VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 uv;

void main(){
    gl_Position = vec4(in_position, 1.0);
    uv =in_texcoord_0;
}

#elif FRAGMENT_SHADER

in vec2 uv;
out vec4 fragColor;


void main(){
    fragColor = vec4(0.1, 0.0, 0.0, 0.0);
}

#endif