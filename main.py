from hmac import new
import moderngl as mgl
import moderngl_window as mglw
from moderngl_window.geometry import quad_fs, quad_2d, cube, sphere
import numpy as np
from struct import unpack
from pyrr import Matrix44 as m44
from moderngl_window.opengl.vao import VAO
import logging

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
    log_level = logging.ERROR
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        
        # ------------------------------------ INITIALIZATION
        self.N = 128
        self.size = (self.N, self.N)
        gs = 16
        self.nxyz = [int(self.N/gs), int(self.N/gs), 1] # layout size
        
        self.x_size = [-1.0, 1.0]
        self.y_size = [-1.0, 1.0]
        self.orto_proj = m44.orthogonal_projection(*self.x_size, *self.y_size, -5, 5)
        self.pers_proj = m44.perspective_projection(70, self.aspect_ratio, 0.1, 40)
        self.orto_view = m44.look_at(np.array([0, 0, 1]), np.array([0, 0, 0]), np.array([0, -1, 0]))
        self.pers_view = m44.look_at(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 0, 1]))

        # ------------------------------------ INTEGRAL REDUCE TEXTURES
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

        # ------------------------- SCENES
        self.quad = quad_fs()
        self.bunny: VAO = self.load_scene("scenes/bunny.obj").meshes[0].vao
        self.monkey: VAO = self.load_scene("scenes/monkey.obj").meshes[0].vao
        self.plane = quad_2d(size=(10, 10), uvs=False)
        self.box = cube()
        self.sph = sphere(0.5)

        # self.bunny = self.sph

        self.bunny_model = m44.from_translation([0.0,0,0])*m44.from_x_rotation(-np.radians(0))
        self.plane_model = m44.from_translation([0,0,-0.6])*m44.from_x_rotation(np.radians(0))
        

        # --------------------------------------------------------------------- PROGRAMS
        self.int_comp = self.load_compute_shader("compute/integral_reduce.comp")
        self.sum_comp = self.load_compute_shader("compute/add_layers.comp")

        self.debug_prog = self.load_program("shaders/debug.glsl")
        self.simple_uv_prog = self.load_program("shaders/simpleUV.glsl")
        self.simple3d_prog = self.load_program("shaders/simple3d.glsl")
        self.peel_prog = self.load_program("shaders/peel.glsl")

        
        # ----------------------------- UNIFORMS
        dx = np.abs(self.x_size[0]-self.x_size[1])/self.N
        dy = np.abs(self.y_size[0]-self.y_size[1])/self.N

        self.peel_prog["size"].value = self.N
        self.peel_prog["dx"].value = dx
        self.peel_prog["dy"].value = dy

        # ---------------- LAYERED TEXTURE FB
        # create texture with size height x n*width
        # every width is a layer
        self.max_layer = 7
        
        # ---------------------- OFF SCREEN FRAMEBUFFER
        self.depth_tex = self.ctx.depth_texture((self.N*self.max_layer, self.N))
        self.tex = self.ctx.texture((self.N*self.max_layer, self.N), components=4, dtype='f4')
        self.tex.filter = mgl.NEAREST, mgl.NEAREST

        self.mass_tex = self.ctx.texture((self.N*self.max_layer, self.N), components=4, dtype='f4')
        self.mass_tex.filter = mgl.NEAREST, mgl.NEAREST

        self.fb = self.ctx.framebuffer(color_attachments=[self.tex, self.mass_tex], depth_attachment=self.depth_tex)
        self.scissor = self.fb.scissor
        self.viewport = self.fb.viewport

        #---------------------- COPY FRAMEBUFFER
        self.copy_tex = self.ctx.texture((self.N*self.max_layer, self.N), components=4, dtype='f4')
        self.copy_tex.filter = mgl.NEAREST, mgl.NEAREST
        self.copy_fb = self.ctx.framebuffer(color_attachments=(self.copy_tex))

        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.disable(mgl.CULL_FACE)

        #--------------------- TEMP FRAMEBUFFER
        self.temp_tex = self.ctx.texture((self.N, self.N), components=4, dtype='f4')
        self.temp_tex.filter = mgl.NEAREST, mgl.NEAREST
        self.temp_fb = self.ctx.framebuffer(color_attachments=(self.temp_tex))

        self.bunny_prop = self.calc_mass_prop(self.bunny, self.bunny_model, self.plane, m44.from_translation([0, 0, 2]))
        print(self.bunny_prop)
        print("ok")

        self.b_pos = np.array([0., 0., 0.])
        self.b_vel = np.array([0., 0., 0.])
        self.b_acc = np.array([0., 0., 0.])


    def calc_mass_prop(self, scene:VAO, scene_model:m44, surf:VAO, surf_model:m44):
        ############### ---------------------------------------------------------------------------------------------- CALC MASS PROP
        # ------ FIRST LAYER
        self.peel_prog["proj"].write(self.orto_proj.astype('f4'))
        self.peel_prog["view"].write(self.orto_view.astype('f4'))
        self.peel_prog["calc_water_surf"].value = 0

        self.fb.clear()
        self.fb.use()
        self.fb.viewport = (0, 0, self.N, self.N)
        self.peel_prog['model'].write(surf_model.astype('f4'))
        surf.render(self.peel_prog)

        # -------------- PEELS
        for i in range(self.max_layer):
            self.peel(i, scene, scene_model)

        # ------------ WATER_SURF
        self.copy_fb.clear()
        self.copy_fb.use()
        self.depth_tex.compare_func=""
        self.depth_tex.use(0)
        self.quad.render(self.debug_prog)
        self.depth_tex.compare_func="<"

        
        self.fb.viewport = (0, 0, self.N, self.N)
        self.fb.scissor = (0, 0, self.N, self.N)
        
        self.fb.use()
        self.fb.clear()
        self.peel_prog['model'].write(surf_model.astype('f4'))
        self.peel_prog["calc_water_surf"].value = 1
        set_uniform(self.peel_prog, [{'nlayers': self.max_layer, 'layer':1, 'clayer':0}, {}])
        self.depth_tex.use(0)
        self.tex.use(1)
        surf.render(self.peel_prog)
        self.fb.scissor = self.scissor
        self.fb.viewport = self.viewport

        # ------------------------------ ADD ALL LAYERS
        self.mass_tex.bind_to_image(0)
        self.temp_tex.bind_to_image(1)
        self.sum_comp['nlayers'].value = self.max_layer
        self.sum_comp.run(*self.nxyz)
        
        # ------------------------ INTEGRATE ALL summed layer

        self.integrated = self.find_sum(self.temp_tex)
        mass_prop = np.array(unpack('ffff', self.integrated), dtype='f4')
        mass_prop[0] /= mass_prop[3] # rx = 1/V | n_z*z*x dxdy
        mass_prop[1] /= mass_prop[3] # ry = 1/V | n_z*z*y dxdy
        mass_prop[2] /= 2*mass_prop[3] # rz = 1/2V | n_z*z*z dxdy

        return mass_prop
        
    def peel(self, layer: int, scene: VAO, model: m44):
        # ---------- PEEL
        self.copy_fb.clear()
        self.copy_fb.use()
        self.depth_tex.compare_func = ""
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
                texIn.bind_to_image(0)
                self.sum_tex_arr[0].bind_to_image(1)
            else:
                self.sum_tex_arr[i-1].bind_to_image(0)
                self.sum_tex_arr[i].bind_to_image(1)

            gs = int(self.N/np.power(2, i+1))
            self.int_comp.run(gs, gs, 1)
        return self.sum_tex_arr[-1].read()


    def render(self, t, dt):
        # ----------------------------------- MASS PROP
        self.rhoL = 1.0
        self.rhoM = 0.6

        mass_prop = self.calc_mass_prop(self.bunny, self.bunny_model, self.plane, self.plane_model)
        m = self.rhoM*self.bunny_prop[3]

        # ----- FORCE CALC
        Fm = m*-9.81  # Body force
        Fb = self.rhoL*mass_prop[3]*9.81 # Buoyancy Force
        in_water = 1 if mass_prop[3] > 0 else 0
        Fd = -in_water*1*self.b_vel*np.abs(self.b_vel) - in_water*0.4*self.b_vel #Drag force  - a*v^2 - b*v  v^2 term for fast motion, v term for quasi static motion  
        F = np.array([0, 0, Fm+Fb])+Fd

        # ------ VERLET INTEGRATION
        # https://en.wikipedia.org/wiki/Verlet_integration
        new_pos = self.b_pos + self.b_vel*dt + self.b_acc*(dt*dt*0.5)
        new_acc = F/m
        new_vel = self.b_vel + (self.b_acc+new_acc)*(dt*0.5)
        
        self.b_pos = new_pos
        self.b_vel = new_vel
        self.b_acc = new_acc


        # ------------ APPLY POSITION
        self.bunny_model = m44.from_translation(self.b_pos)
        self.wnd.fbo.use()
        
        self.ctx.disable(mgl.CULL_FACE)
        self.ctx.enable(mgl.DEPTH_TEST)

        # ----------------------------------------
        self.wnd.fbo.viewport = (0, 0, *self.window_size)
        self.simple3d_prog['view'].write(self.pers_view.astype('f4'))
        self.simple3d_prog['proj'].write(self.pers_proj.astype('f4'))

        self.simple3d_prog['model'].write(self.plane_model.astype('f4'))
        self.plane.render(self.simple3d_prog)
        self.simple3d_prog['model'].write(self.bunny_model.astype('f4'))
        self.bunny.render(self.simple3d_prog)


        self.wnd.fbo.viewport = (0, 0, self.max_layer*self.N, self.N)
        self.tex.use(0)
        self.quad.render(self.debug_prog)

        self.wnd.fbo.viewport = (self.window_size[0]-self.N*2, 0, 2*self.N, 2*self.N)
        self.temp_tex.use()
        self.quad.render(self.debug_prog)
        

Window.run()