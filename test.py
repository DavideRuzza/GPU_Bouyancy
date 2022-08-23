import moderngl as mgl
import moderngl_window as mglw
from moderngl_window.geometry import quad_fs, quad_2d
from numpy import dtype
from pyrr import Matrix44 as m44
from moderngl_window.opengl.vao import VAO

class Surface:

    def __init__(self, ctx, size, dtype='f4'):
        self.ctx = ctx
        self.size = size
        self.dtype = dtype

        self.texture : mgl.Texture = self.tex()
        self.depth_texture : mgl.Texture = self.ctx.depth_texture(size=(self.size, self.size))
        self.fb: mgl.framebuffer.Framebuffer = self.ctx.framebuffer(color_attachments=(self.texture), depth_attachment=(self.depth_texture))

    def tex(self):
        tex = self.ctx.texture((self.size, self.size), components=4, dtype=self.dtype)
        tex.filter = mgl.NEAREST, mgl.NEAREST
        tex.repeat_x = False
        tex.repeat_y = False
        return tex


class Window(mglw.WindowConfig):

    window_size=(512, 512)
    title="Test"
    gl_version=(4, 6)
    resource_dir="resources"
    aspect_ratio=1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.N = 256
        gs = 16
        self.nxyz = [int(self.N/gs), int(self.N/gs), 1]

        # ------------------------- SCENES
        self.quad = quad_fs()
        self.plane = quad_2d((2, 2), uvs=False)
        self.bunny: VAO = self.load_scene("scenes/bunny.obj").meshes[0].vao
        self.monkey: VAO = self.load_scene("scenes/monkey.obj").meshes[0].vao

        # -------------------------- FB

        self.depthTex = self.ctx.depth_texture((self.N, self.N))
        self.colorTex = self.ctx.texture((self.N, self.N), components=4, dtype='f4')
        self.test_fb = self.ctx.framebuffer(color_attachments=[self.colorTex], depth_attachment=self.depthTex)

        self.copy_surf = Surface(self.ctx, self.N)
        # self.colorTex2 = self.ctx.texture((self.N, self.N), components=4, dtype='f4')

        # -------------------------- PROGRAMS   
        self.debug_prog = self.load_program("shaders/debug.glsl")
        self.simp_uv_prog = self.load_program("shaders/simpleUV.glsl")
        self.simp3d_prog = self.load_program("shaders/simple3d.glsl")
        self.peel_prog = self.load_program("shaders/peel.glsl")

        # -------------------------- PROJECTION
        near, far = -4, 4
        proj = m44.orthogonal_projection(-1, 1, -1, 1, near, far)
        view = m44.look_at([0, 1, 0], [0, 0, 0], [0, 0, 1])
        model = m44.from_x_rotation(90)
        pv = proj*view # projection view

        # -------------------------- UNIFORMS
        self.simp3d_prog["pv"].write(pv.astype('f4'))
        self.simp3d_prog["m"].write(model.astype('f4'))
        self.peel_prog["pv"].write(pv.astype('f4'))
        self.peel_prog["m"].write(model.astype('f4'))


        # ----------------------------------- PRE RENDER

    def copy_depth(self):
        self.copy_surf.fb.clear()
        self.copy_surf.fb.use()
        self.depthTex.compare_func = ""
        self.depthTex.use(0)
        self.quad.render(self.debug_prog)
        self.depthTex.compare_func = "<"

    def render(self, t, dt):
        

        self.ctx.disable(mgl.CULL_FACE)
        self.ctx.enable(mgl.DEPTH_TEST)

        # -----step 1 draw object
        self.test_fb.clear()
        self.test_fb.use()
        self.plane.render(self.simp3d_prog)

        # ---------- peel
        for i in range(6):
            self.copy_depth()
            self.test_fb.clear()
            self.test_fb.use()
            self.copy_surf.texture.use(0)
            self.bunny.render(self.peel_prog)

        self.wnd.fbo.use()
        self.ctx.clear()
        
        self.wnd.fbo.viewport = (0, 256, 256, 256)
        self.colorTex.use(0)
        self.quad.render(self.debug_prog)

        self.wnd.fbo.viewport = (256, 256, 256, 256)
        self.depthTex.use(0)
        self.quad.render(self.debug_prog)

        self.wnd.fbo.viewport = (0, 0, 256, 256)
        self.copy_surf.texture.use(0)
        self.quad.render(self.debug_prog)
        




Window.run()