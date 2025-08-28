try:
    import bpy, bmesh

    inside_blender: bool = True
except ModuleNotFoundError:
    inside_blender: bool = False

from pathlib import Path

if not inside_blender:

    def ambient_occlusion_map(
        input_height_map_path: Path,
        input_opacity_map_path: Path,
        output_ambient_occlusion_map_path: Path,
        blender_path: Path,
    ) -> None:
        import subprocess

        subprocess.check_output(
            [
                "sudo",
                str(blender_path),
                "--background",
                "-noaudio",
                "--python",
                # TODO Get this path dynamically.
                "./photometric_stereo_mappings/mappings/ambient_occlusion.py",
                "--",
                str(input_height_map_path),
                str(input_opacity_map_path),
                str(output_ambient_occlusion_map_path),
            ]
        )


if inside_blender:
    import sys

    argv = sys.argv
    parameters = argv[argv.index("--") + 1 :]  # get all args after "--"

    height_map_path = Path(parameters[0])
    opacity_map_path = Path(parameters[1])
    ambient_occlusion_map_path = Path(parameters[2])

    def blender_bake_ambient_occlusion_map(
        height_map_path: Path,
        opacity_map_path: Path,
        output_ambient_occlusion_map_path: Path,
        subdivision_factor: float = 1,
    ) -> None:
        """Bakes ray-traced ambient occlusion map using blender cycles as a render engine.

        Mainly written by Biman.

        Parameters
        ----------
        height_map_path : Path
            Path to the height map.
        opacity_map_path : Path
            Path to the opacity map.
        output_ambient_occlusion_map_path : Path
            Path to the output ambient occlusion map.
        subdivision_factor : float, optional
            The subdivision factor. Values higher than 1 do not make sense. The default is 1.
        """

        # Delete all objects
        for ob in bpy.data.objects:
            bpy.data.objects.remove(ob)

        # Import the height map
        height_map_image = bpy.data.images.load(str(height_map_path))
        height_map_image.colorspace_settings.name = "Non-Color"

        output_width, output_height = height_map_image.size

        # Import opacity map
        opacity_image = bpy.data.images.load(str(opacity_map_path))
        opacity_image.colorspace_settings.name = "Non-Color"

        ao_image = bpy.data.images.new("AO", output_width, output_height)

        # Create the plane
        me = bpy.data.meshes.new("Plane")
        bm = bmesh.new()
        bm.loops.layers.uv.new("Plane")
        bmesh.ops.create_grid(
            bm,
            size=1,
            x_segments=round(output_width * subdivision_factor),
            y_segments=round(output_height * subdivision_factor),
            calc_uvs=True,
        )
        bm.to_mesh(me)
        ob = bpy.data.objects.new("Plane", me)
        bpy.context.collection.objects.link(ob)

        # Add a material
        mat = bpy.data.materials.new("ao")
        mat.use_nodes = True
        mat.cycles.displacement_method = "DISPLACEMENT"
        nodes = mat.node_tree.nodes

        nodes.clear()

        # Create ao image node
        ao_node = nodes.new("ShaderNodeTexImage")
        ao_node.image = ao_image
        ao_node.select = True

        # Create height map image node
        image_tex_node = nodes.new("ShaderNodeTexImage")
        image_tex_node.image = height_map_image
        image_tex_node.select = False

        # Create material output node
        material_output_node = nodes.new("ShaderNodeOutputMaterial")
        material_output_node.select = False

        # Create displacement node
        displacement_node = nodes.new("ShaderNodeDisplacement")
        displacement_node.select = False

        # Connect nodes
        mat.node_tree.links.new(
            displacement_node.inputs[0], image_tex_node.outputs["Color"]
        )
        mat.node_tree.links.new(
            material_output_node.inputs[2], displacement_node.outputs["Displacement"]
        )

        # Add the material to object
        ob.data.materials.append(mat)

        # Delete masked vertices using opacity map
        mask_group = ob.vertex_groups.new(name="mask_group")
        texture = bpy.data.textures.new("texture", type="IMAGE")
        texture.image = opacity_image

        vertex_mix_mod = ob.modifiers.new(
            name="AO_vertex_mix", type="VERTEX_WEIGHT_MIX"
        )
        vertex_mix_mod.vertex_group_a = "mask_group"
        vertex_mix_mod.default_weight_b = 1
        vertex_mix_mod.mix_set = "ALL"
        vertex_mix_mod.mask_texture = texture
        vertex_mix_mod.mask_tex_use_channel = "INT"
        vertex_mix_mod.mask_tex_mapping = "LOCAL"

        mask_mod = ob.modifiers.new(name="AO_mask", type="MASK")
        mask_mod.vertex_group = "mask_group"

        # Apply the modifiers
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.modifier_apply(modifier="AO_vertex_mix")
        bpy.ops.object.modifier_apply(modifier="AO_mask")

        # Set scene properties
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.bake_type = "AO"
        bpy.context.scene.cycles.device = "CPU"

        # Make the object active and selected
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob

        # Bake the image
        bpy.ops.object.bake(type="AO")

        ao_image.filepath_raw = str(output_ambient_occlusion_map_path)
        ao_image.file_format = "PNG"
        ao_image.save()

    blender_bake_ambient_occlusion_map(
        height_map_path,
        opacity_map_path,
        ambient_occlusion_map_path,
        subdivision_factor=0.75,
    )
