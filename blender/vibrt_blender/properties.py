import bpy
import time

from . import shader_precompile


def _redraw_properties(context):
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "PROPERTIES":
                area.tag_redraw()


class VIBRT_OT_precompile_shaders(bpy.types.Operator):
    bl_idname = "vibrt.precompile_shaders"
    bl_label = "Precompile Shaders"
    bl_description = (
        "Prepare shaders for this scene and view layer before rendering. "
        "Changing shader nodes may require another compilation"
    )
    _timer = None

    @classmethod
    def poll(cls, context):
        return (
            context.engine == "VIBRT"
            and shader_precompile.active_job is None
            and not bpy.app.is_job_running("RENDER")
        )

    def execute(self, context):
        try:
            self._job = shader_precompile.start(context.scene, context.view_layer)
            if bpy.app.background:
                while not self._job.done():
                    time.sleep(0.1)
                return self._finish(context)
            self._timer = context.window_manager.event_timer_add(
                0.2, window=context.window
            )
            context.window_manager.modal_handler_add(self)
            _redraw_properties(context)
            return {"RUNNING_MODAL"}
        except Exception as exc:
            shader_precompile.stop()
            shader_precompile.last_message = "Shader compilation failed"
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except BaseException:
            shader_precompile.stop()
            raise

    def modal(self, context, event):
        if event.type == "ESC":
            self._job.cancel()
        if self._job.cancelled or self._job.done():
            return self._finish(context)
        if event.type == "TIMER":
            shader_precompile.last_message = self._job.status()
            _redraw_properties(context)
        return {"PASS_THROUGH"}

    def _finish(self, context):
        try:
            if self._job.cancelled:
                shader_precompile.last_message = "Shader compilation cancelled"
                return {"CANCELLED"}
            self._job.result()
            elapsed = time.monotonic() - self._job.started
            shader_precompile.last_message = f"Last compilation: {elapsed:.1f}s"
            self.report({"INFO"}, f"vibrt: shaders compiled in {elapsed:.1f}s")
            return {"FINISHED"}
        except Exception as exc:
            shader_precompile.last_message = "Shader compilation failed"
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            self._job.close()
            if shader_precompile.active_job is self._job:
                shader_precompile.active_job = None
            if self._timer is not None:
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None
            _redraw_properties(context)

    def cancel(self, context):
        self._job.cancel()
        self._finish(context)


class VIBRT_OT_cancel_precompile(bpy.types.Operator):
    bl_idname = "vibrt.cancel_precompile"
    bl_label = "Cancel Compilation"
    bl_description = "Stop the background shader compilation"

    def execute(self, context):
        shader_precompile.stop()
        _redraw_properties(context)
        return {"FINISHED"}


class VIBRT_PT_shaders(bpy.types.Panel):
    bl_label = "Shaders"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    COMPAT_ENGINES = {"VIBRT"}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        job = shader_precompile.active_job
        if job is None:
            layout.operator("vibrt.precompile_shaders", icon="SHADING_RENDERED")
        else:
            layout.operator("vibrt.cancel_precompile", icon="CANCEL")
            layout.label(text=f"Elapsed: {time.monotonic() - job.started:.0f}s")
        if shader_precompile.last_message:
            layout.label(text=shader_precompile.last_message)


@bpy.app.handlers.persistent
def _cancel_compilation_on_load(_):
    shader_precompile.stop()


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
    bpy.utils.register_class(VIBRT_OT_precompile_shaders)
    bpy.utils.register_class(VIBRT_OT_cancel_precompile)
    bpy.utils.register_class(VIBRT_PT_shaders)
    bpy.app.handlers.load_pre.append(_cancel_compilation_on_load)


def unregister():
    shader_precompile.stop()
    bpy.app.handlers.load_pre.remove(_cancel_compilation_on_load)
    bpy.utils.unregister_class(VIBRT_PT_shaders)
    bpy.utils.unregister_class(VIBRT_OT_cancel_precompile)
    bpy.utils.unregister_class(VIBRT_OT_precompile_shaders)
    bpy.utils.unregister_class(VIBRT_PT_denoising)
    bpy.utils.unregister_class(VIBRT_PT_sampling)
    del bpy.types.Scene.vibrt_spp
    del bpy.types.Scene.vibrt_clamp_indirect
    del bpy.types.Scene.vibrt_denoise
