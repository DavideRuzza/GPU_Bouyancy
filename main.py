import moderngl as mgl
import moderngl_window as mglw
from moderngl_window.geometry import quad_fs
from moderngl_window.geometry import quad_2d
import numpy as np
from struct import unpack
import pyrr
from pyrr import Matrix44 as m44
from moderngl_window.opengl.vao import VAO

class Surface:

    def __init__(self, ctx, size, dtype='f4'):
        self.ctx: mgl.Context = ctx
        self.size = size
        self.dtype = dtype

        self.texture : mgl.Texture = self.tex()
        self.depth_texture :mgl.Texture = self.ctx.depth_texture(self.size)
        self.fb:mgl.Framebuffer = self.ctx.framebuffer(color_attachments=(self.texture), depth_attachment=(self.depth_texture))

    def tex(self):
        tex = self.ctx.texture(self.size, components=4, dtype=self.dtype)
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
    """ [to_value, to_write]"""
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
    window_size = (2*512, 512)
    resizable = False
    resource_dir = "resources/"
    aspect_ratio = window_size[0]/window_size[1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ------------------------------------ INITIALIZATION
        self.N = 256
        self.size = (self.N, self.N)
        gs = 16
        self.nxyz = [int(self.N/gs), int(self.N/gs), 1] # layout size
        
        self.orto_proj = m44.orthogonal_projection(-1, 1, 1, -1, -3, 3)
        self.pers_proj = m44.perspective_projection(70, self.aspect_ratio, 0.1, 40)
        self.orto_view = m44.look_at(np.array([0, 0, 1]), np.array([0, 0, 0]), np.array([0, -1, 0]))
        self.pers_view = m44.look_at(np.array([2, 0, 0.4]), np.array([0, 0, 0]), np.array([0, 0, 1]))

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
        
        # --------------------------------------------------------------------- PROGRAMS

        self.sum_comp = self.load_compute_shader("compute/sumreduce.comp")
        self.debug_prog = self.load_program("shaders/debug.glsl")
        self.simple_uv_prog = self.load_program("shaders/simpleUV.glsl")
        self.simple3d_prog = self.load_program("shaders/simple3d.glsl")
        self.peel_prog = self.load_program("shaders/peel.glsl")

        
        # ----------------------------- UNIFORMS
        self.simple3d_prog["proj"].write((self.orto_proj).astype('f4'))
        self.peel_prog["proj"].write((self.orto_proj).astype('f4'))

        self.simple3d_prog["view"].write((self.orto_view).astype('f4'))
        self.peel_prog["view"].write((self.orto_view).astype('f4'))

        self.peel_prog["size"].value = self.N
        self.peel_prog["calc_water_surf"].value = 0

        # ------------------------- SCENES
        self.quad = quad_fs()
        self.bunny: VAO = self.load_scene("scenes/bunny.obj").meshes[0].vao
        self.monkey: VAO = self.load_scene("scenes/monkey.obj").meshes[0].vao
        self.plane = quad_2d(size=(3, 3), uvs=False)

        self.bunny_model = m44.from_x_rotation(-np.radians(0))
        self.plane_model = m44.from_translation([0,0,0.3])*m44.from_x_rotation(np.radians(0))

        # ------------- SUM TEST
        # self.test_surf = Surface(self.ctx, (self.N*2, self.N*2), 'f4')
        # self.test_surf.fb.clear()
        # self.test_surf.fb.viewport = (0, 0, self.N*2, self.N*2)
        # self.test_surf.fb.use()
        # self.quad.render(self.simple_uv_prog)
        # self.test_sum = self.find_sum(self.test_surf.texture)
        # print(unpack('ffff', self.test_sum))
        # self.wnd.fbo.use()

        # ---------------- LAYERED TEXTURE FB
        # create texture with size height x n*width
        # every width is a layer
        self.max_layer = 4
        
        # ---------------------- OFF SCREEN FRAMEBUFFER
        self.depth_tex = self.ctx.depth_texture((self.N*self.max_layer, self.N))
        self.tex = self.ctx.texture((self.N*self.max_layer, self.N), components=4, dtype='f4')
        self.tex.filter = mgl.NEAREST, mgl.NEAREST
        self.fb = self.ctx.framebuffer(color_attachments=[self.tex], depth_attachment=self.depth_tex)
        self.scissor = self.fb.scissor

        #---------------------- COPY FRAMEBUFFER
        self.copy_tex = self.ctx.texture((self.N*self.max_layer, self.N), components=4, dtype='f4')
        self.copy_tex.filter = mgl.NEAREST, mgl.NEAREST
        self.copy_fb = self.ctx.framebuffer(color_attachments=(self.copy_tex))

        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.disable(mgl.CULL_FACE)

        # ------ FIRST LAYER

        self.fb.clear(0, 0, 0)
        self.fb.use()
        self.fb.viewport = (0, 0, self.N, self.N)
        self.peel_prog['model'].write(self.plane_model.astype('f4'))
        self.plane.render(self.peel_prog)

        # -------------- PEELS
        for i in range(self.max_layer):
            self.peel(i, self.bunny, self.bunny_model)

        # ------------ WATER_SURF
        self.copy_fb.clear()
        self.copy_fb.use()
        self.depth_tex.compare_func=""
        self.depth_tex.use(0)
        self.quad.render(self.debug_prog)
        self.depth_tex.compare_func="<"
        
        self.fb.use()
        self.fb.viewport = (0, 0, self.N, self.N)
        self.fb.scissor = (0, 0, self.N, self.N)
        self.fb.clear()
        self.peel_prog['model'].write(self.plane_model.astype('f4'))
        self.peel_prog["calc_water_surf"].value = 1
        set_uniform(self.peel_prog, [{'nlayers': self.max_layer, 'layer':1, 'clayer':0}, {}])
        self.depth_tex.use(0)
        self.tex.use(1)
        self.plane.render(self.peel_prog)
        self.fb.scissor = self.scissor


    def peel(self, layer: int, scene: VAO, model: m44):
        # ---------- PEEL
        self.copy_fb.clear()
        self.copy_fb.use()
        self.depth_tex.compare_func=""
        self.depth_tex.use(0)
        self.quad.render(self.debug_prog)
        self.depth_tex.compare_func="<"
        
        self.fb.use()
        self.fb.viewport = (layer*self.N, 0, self.N, self.N)
        set_uniform(self.peel_prog, [{'nlayers': self.max_layer, 'layer':layer, 'clayer':layer+1}, {}])
        self.peel_prog['model'].write(model.astype('f4'))
        self.copy_tex.use(0)
        scene.render(self.peel_prog)

    # -----------
    def find_sum(self, texIn: mgl.Texture):
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
        
        self.ctx.clear(0.1, 0.1, 0.1)
        self.wnd.fbo.use()
        # self.ctx.disable(mgl.CULL_FACE)
        # self.ctx.disable(mgl.DEPTH_TEST)

        # ----------------------------------------
        self.wnd.fbo.viewport = (0, 0,self.max_layer*self.N, self.N)
        self.tex.use(0)
        self.quad.render(self.debug_prog)
        

Window.run()
"""

import moderngl as mgl
import moderngl_window as mglw
from moderngl_window.geometry import quad_fs
from moderngl_window.geometry import quad_2d
import numpy as np
from struct import unpack
import pyrr
from pyrr import Matrix44 as m44
from moderngl_window.opengl.vao import VAO

class Surface:

    def __init__(self, ctx, size, dtype='f4'):
        self.ctx: mgl.Context = ctx
        self.size = size
        self.dtype = dtype

        self.texture : mgl.Texture = self.tex()
        self.depth_texture :mgl.Texture = self.ctx.depth_texture(self.size)
        self.fb:mgl.Framebuffer = self.ctx.framebuffer(color_attachments=(self.texture), depth_attachment=(self.depth_texture))

    def tex(self):
        tex = self.ctx.texture(self.size, components=4, dtype=self.dtype)
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
    ''' [to_value, to_write]'''
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
    window_size = (2*512, 512)
    resizable = False
    resource_dir = "resources/"
    aspect_ratio = window_size[0]/window_size[1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ------------------------------------ INITIALIZATION
        self.N = 256
        self.size = (self.N, self.N)
        gs = 16
        self.nxyz = [int(self.N/gs), int(self.N/gs), 1] # layout size
        
        self.orto_proj = m44.orthogonal_projection(-1, 1, -1, 1, -3, 3, dtype='f4')
        self.pers_proj = m44.perspective_projection(70, self.aspect_ratio, 0.1, 40, dtype='f4')
        self.orto_view = m44.look_at(np.array([0, 0, 1]), np.array([0, 0, 0]), np.array([0, 1, 0]), dtype='f4')
        self.pers_view = m44.look_at(np.array([2, 0, 0.4]), np.array([0, 0, 0]), np.array([0, 0, 1]), dtype='f4')
        

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
        
        # --------------------------------------------------------------------- PROGRAMS

        self.sum_comp = self.load_compute_shader("compute/sumreduce.comp")
        self.debug_prog = self.load_program("shaders/debug.glsl")
        self.simple_uv_prog = self.load_program("shaders/simpleUV.glsl")
        self.simple3d_prog = self.load_program("shaders/simple3d.glsl")
        self.peel_prog = self.load_program("shaders/peel.glsl")

        
        # ----------------------------- UNIFORMS
        self.simple3d_prog['projection'].write(self.orto_proj)
        self.simple3d_prog['view'].write(self.orto_view)
        self.peel_prog['projection'].write(self.orto_proj)
        self.peel_prog['size'].value = self.N
        self.peel_prog['calc_water_surf'].value = 0

        # ------------------------- SCENES
        self.quad = quad_fs()
        self.bunny: VAO = self.load_scene("scenes/bunny.obj").meshes[0].vao
        self.monkey: VAO = self.load_scene("scenes/monkey.obj").meshes[0].vao
        self.plane = quad_2d(size=(10, 10), uvs=False)

        self.bunny_model = m44.from_x_rotation(-np.radians(0), dtype='f4')
        self.plane_model = m44.from_translation([0,0, 0.0, 0.0], dtype='f4')*m44.from_x_rotation(np.radians(0), dtype='f4')

        # ------------- SUM TEST
        # self.test_surf = Surface(self.ctx, (self.N*2, self.N*2), 'f4')
        # self.test_surf.fb.clear()
        # self.test_surf.fb.viewport = (0, 0, self.N*2, self.N*2)
        # self.test_surf.fb.use()
        # self.quad.render(self.simple_uv_prog)
        # self.test_sum = self.find_sum(self.test_surf.texture)
        # print(unpack('ffff', self.test_sum))
        # self.wnd.fbo.use()

        # ---------------- LAYERED TEXTURE FB
        # create texture with size height x n*width
        # every width is a layer
        self.max_layer = 3
        
        # ---------------------- OFF SCREEN FRAMEBUFFER
        self.depth_tex = self.ctx.depth_texture((self.N*self.max_layer, self.N))
        self.tex = self.ctx.texture((self.N*self.max_layer, self.N), components=4, dtype='f4')
        self.tex.filter = mgl.NEAREST, mgl.NEAREST
        self.fb = self.ctx.framebuffer(color_attachments=[self.tex], depth_attachment=self.depth_tex)
        self.scissor = self.fb.scissor

        #---------------------- COPY FRAMEBUFFER
        self.copy_tex = self.ctx.texture((self.N*self.max_layer, self.N), components=4, dtype='f4')
        self.copy_tex.filter = mgl.NEAREST, mgl.NEAREST
        self.copy_fb = self.ctx.framebuffer(color_attachments=(self.copy_tex))



        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.disable(mgl.CULL_FACE)

        # ------------------------------------------------------- PEEL PROCEDURE
        # ------ FIRST LAYER
        self.peel_prog['projection'].write(self.orto_proj)
        self.peel_prog['view'].write(self.orto_view)
        self.simple3d_prog['projection'].write(self.orto_proj)
        self.simple3d_prog['view'].write(self.orto_view)
        
        self.fb.clear(0, 0, 0)
        self.fb.use()
        self.fb.viewport = (0, 0, self.N, self.N)
        self.peel_prog['model'].write(self.plane_model.astype('f4'))
        set_uniform(self.peel_prog, [{'nlayers': self.max_layer, 'layer':0, 'clayer':0}, {}])
        self.plane.render(self.peel_prog)

        # ---------- PEEL
        self.copy_fb.clear()
        self.copy_fb.use()
        self.depth_tex.compare_func=""
        self.depth_tex.use(0)
        self.quad.render(self.debug_prog)
        self.depth_tex.compare_func="<"
        
        self.fb.use()
        self.fb.viewport = (1*self.N, 0, self.N, self.N)
        set_uniform(self.peel_prog, [{'nlayers': self.max_layer, 'layer':0, 'clayer':1}, {}])
        self.peel_prog['model'].write(self.bunny_model.astype('f4'))
        self.copy_tex.use(0)
        self.bunny.render(self.peel_prog)

        # -------------- PEELS
        # for i in range(self.max_layer):
        #     self.peel(i, self.bunny, self.bunny_model)

        # ------------ WATER_SURF
        self.copy_fb.clear()
        self.copy_fb.use()
        self.depth_tex.compare_func=""
        self.depth_tex.use(0)
        self.quad.render(self.debug_prog)
        self.depth_tex.compare_func="<"
        
        self.fb.use()
        self.fb.viewport = (0, 0, self.N, self.N)
        self.fb.scissor = (0, 0, self.N, self.N)
        self.fb.clear()
        self.peel_prog['model'].write(self.plane_model.astype('f4'))
        self.peel_prog["calc_water_surf"].value = 1
        set_uniform(self.peel_prog, [{'nlayers': self.max_layer, 'layer':1, 'clayer':0}, {}])
        self.depth_tex.use(0)
        self.tex.use(1)
        self.plane.render(self.peel_prog)
        self.fb.scissor = self.scissor # reset scissor


    def peel(self, layer: int, scene: VAO, model: m44):
        # ---------- PEEL
        self.copy_fb.clear()
        self.copy_fb.use()
        self.depth_tex.compare_func=""
        self.depth_tex.use(0)
        self.quad.render(self.debug_prog)
        self.depth_tex.compare_func="<"
        
        self.fb.use()
        self.fb.viewport = (layer*self.N, 0, self.N, self.N)
        set_uniform(self.peel_prog, [{'nlayers': self.max_layer, 'layer':layer, 'clayer':layer+1}, {}])
        self.peel_prog['model'].write(model.astype('f4'))
        self.copy_tex.use(0)
        scene.render(self.peel_prog)


        
    # -----------
    def find_sum(self, texIn: mgl.Texture):
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
        
        self.ctx.clear(0.1, 0.1, 0.1)
        self.wnd.fbo.use()

        # self.ctx.enable(mgl.DEPTH_TEST)
        # self.simple3d_prog['projection'].write(self.pers_proj)
        # self.simple3d_prog['view'].write(self.pers_view)
        # self.simple3d_prog['model'].write(self.bunny_model)
        # self.bunny.render(self.simple3d_prog)
        # self.simple3d_prog['model'].write(self.plane_model.astype('f4'))
        # self.plane.render(self.simple3d_prog)
        

        # self.ctx.disable(mgl.DEPTH_TEST)
        # ----------------------------------------
        # self.wnd.fbo.viewport = (0, 0, self.max_layer*self.N, self.N)
        self.wnd.fbo.viewport = (0, 0, 2*512, 512)
        self.tex.use(0)
        self.quad.render(self.debug_prog)

        

Window.run()

"""