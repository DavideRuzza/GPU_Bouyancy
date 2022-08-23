#version 460

uniform mat4 pv;
uniform mat4 m;

uniform float nlayers; // numbers of layers
uniform float layer;   // layer to view
uniform float clayer;  // current layer

layout(binding=0) uniform sampler2D depthTex;

#ifdef VERTEX_SHADER

in vec3 in_position;
in vec3 in_normal;

out vec3 N;
out vec3 pos;

void main(){
    gl_Position = pv*m*vec4(in_position, 1.0);
    N = normalize(in_normal);
}

#elif FRAGMENT_SHADER

in vec3 N;
layout(location = 0) out vec4 fragOut;

void main(){
    vec2 UV = gl_FragCoord.xy/128.;
    UV.x = UV.x/nlayers - (clayer-layer)/nlayers;
    fragOut = texture(depthTex, UV);

}

#endif
