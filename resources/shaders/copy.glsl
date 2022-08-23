#version 460


#ifdef VERTEX_SHADER

void main(){
    gl_Position = in_position;
    uv = in_texcoors_0;
}

#elif FRAGMENT_SHADER

void main(){

}

#endif