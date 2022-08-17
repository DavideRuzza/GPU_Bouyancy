import moderngl as mgl
import moderngl_window as mglw
from moderngl_window.geometry import quad_fs
import numpy as np
from struct import unpack

class Surface:

    def __init__(self, ctx, size, dtype='f2'):
        self.ctx = ctx
        self.size = size
        self.dtype = dtype

        self.texture:mgl.Texture = self.tex()
        self.fb = self.ctx.framebuffer(color_attachments=(self.texture))

    def tex(self):
        tex = self.ctx.texture((self.size, self.size), components=4, dtype=self.dtype)
        tex.filter = mgl.NEAREST, mgl.NEAREST
        tex.repeat_x = False
        tex.repeat_y = False
        return tex

class Slab:
    
    def __init__(self, ctx, size, dtype='f2'):

        self.ping = Surface(ctx, size, dtype)
        self.pong = Surface(ctx, size, dtype)
    
    def swap(self):
        self.ping.texture, self.pong.texture = self.pong.texture, self.ping.texture
        self.ping.fb, self.pong.fb =  self.pong.fb, self.ping.fb


def set_uniform(program:mgl.Program, uniforms: list):
    if len(uniforms) != 2:
        raise Exception("value and write must be defined")
    
    to_value = uniforms[0]
    to_write = uniforms[1]

    for val in to_value:
        try:
            program[val].value = to_value[val]
        except:
            pass
    
    for val in to_write:
        try:
            program[val].write(to_write[val])
        except:
            pass

class Window(mglw.WindowConfig):
    gl_version = (4, 6)
    window_size = (512, 512)
    resizable = False
    resource_dir = "resources/"
    aspect_ratio = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ------------------------------------ INITIALIZATION
        self.N = 64
        self.size = (self.N, self.N)
        gs = 8
        self.nxyz = [int(self.N/gs), int(self.N/gs), 1] # layout size
        
        self.simple_prog = self.load_program("shaders/simple.glsl")
        self.simple_col_prog = self.load_program("shaders/simple_col.glsl")

        self.quad = quad_fs()

        # ------------------------------------ SUM REDUCE
        self.log2N = int(np.log2(self.N))

        self.sum_tex_arr=[]
        
        for i in range(self.log2N):
            tex = self.ctx.texture(
            (
                int(self.N/np.power(2, i+1)),
                int(self.N/np.power(2, i+1))
            ), 4, dtype='f4')
            tex.filter = mgl.NEAREST, mgl.NEAREST
            tex.repeat_x = False
            tex.repeat_y = False
            self.sum_tex_arr.append(tex)
        
        self.sum_comp = self.load_compute_shader("compute/sumreduce.comp")

        # ------------- TEST INTEGRAL
        self.test_surf = Surface(self.ctx, self.N*2, 'f4')
        self.test_surf.fb.clear()
        self.test_surf.fb.viewport = (0, 0, self.N*2, self.N*2)
        self.test_surf.fb.use()
        self.quad.render(self.simple_col_prog)
        

        self.test_sum = self.find_sum(self.test_surf.texture)
        print(unpack('ffff', self.test_sum))

        self.wnd.fbo.use()



    # -----------
    def find_sum(self, texIn):
        # find integral of texture summ all values
        iters = self.log2N

        for i in range(iters):
            if i == 0:
                print('0')
                texIn.bind_to_image(0)
                self.sum_tex_arr[0].bind_to_image(1)
            else:
                self.sum_tex_arr[i-1].bind_to_image(0)
                self.sum_tex_arr[i].bind_to_image(1)

            gs = int(self.N/np.power(2, i+1))
            self.sum_comp.run(gs, gs, 1)
        return self.sum_tex_arr[-1].read()

    
    

    def render(self, t, dt):
        self.ctx.clear(0, 0, 0.4)
        # self.ctx.fbo.viewport=(0, 0, 64, 64)
        # self.test_surf.texture.use(0)
        # self.quad.render(self.simple_prog)

        # for i, tex in enumerate(self.sum_tex_arr):
            
        #     self.ctx.fbo.viewport=((i+1)*64, 0, 64, 64)
        #     tex.use(0)
        #     self.quad.render(self.simple_prog)
        

Window.run()
