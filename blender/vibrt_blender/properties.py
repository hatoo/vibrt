import bpy


class VIBRT_PT_sampling(bpy.types.Panel):
    bl_label = "Sampling"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    COMPAT_ENGINES = {"VIBRT"}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(context.scene, "vibrt_spp")
        layout.prop(context.scene, "vibrt_clamp_indirect")


class VIBRT_PT_denoising(bpy.types.Panel):
    bl_label = "Denoising"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    COMPAT_ENGINES = {"VIBRT"}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw_header(self, context):
        self.layout.prop(context.scene, "vibrt_denoise", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.active = context.scene.vibrt_denoise
        layout.label(text="OptiX AI denoiser (AOV model, albedo + normal guides)")


def register():
    bpy.types.Scene.vibrt_spp = bpy.props.IntProperty(
        name="Samples",
        description="Samples per pixel for vibrt rendering",
        default=64,
        min=1,
        soft_max=4096,
    )
    bpy.types.Scene.vibrt_clamp_indirect = bpy.props.FloatProperty(
        name="Clamp Indirect",
        description="Clamp indirect (bounce>=1) contribution luminance. 0 disables",
        default=10.0,
        min=0.0,
        soft_max=100.0,
    )
    bpy.types.Scene.vibrt_denoise = bpy.props.BoolProperty(
        name="Denoise",
        description="Run the OptiX AI denoiser on the final image",
        default=False,
    )
    bpy.utils.register_class(VIBRT_PT_sampling)
    bpy.utils.register_class(VIBRT_PT_denoising)


def unregister():
    bpy.utils.unregister_class(VIBRT_PT_denoising)
    bpy.utils.unregister_class(VIBRT_PT_sampling)
    del bpy.types.Scene.vibrt_spp
    del bpy.types.Scene.vibrt_clamp_indirect
    del bpy.types.Scene.vibrt_denoise
