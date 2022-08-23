#version 460

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;

uniform float nlayers; // numbers of layers
uniform float layer;   // layer to view
uniform float clayer;  // current layer
uniform float size;
uniform int calc_water_surf;

layout(binding=0) uniform sampler2D depthTex;
layout(binding=1) uniform sampler2D normTex;

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
layout(location = 0) out vec4 fragOut;

void main(){

    if (int(clayer)==0 && calc_water_surf==0){
        fragOut = vec4(N, 1.0);
    } else {

        vec2 UV = gl_FragCoord.xy/size;
        UV.x = UV.x/nlayers - (clayer-layer)/nlayers;// + (nlayers-layer)/nlayers;
        
        if (calc_water_surf==1){
            vec3 normal = texture(normTex, UV).xyz;
            float depth = texture(depthTex, UV).r;
            if (normal.z > 0 || depth > 0.99) {
                discard;
            }
            fragOut = vec4(N, 1.0);
            
        } else {

            float depth = texture(depthTex, UV).r;

            if (gl_FragCoord.z<=depth){
                discard;
            }
            fragOut = vec4(N, 1.0);
        }

    }

}

#endif
