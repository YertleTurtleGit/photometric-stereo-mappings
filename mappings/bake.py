try:
    import bpy, bmesh

    inside_blender: bool = True
except ModuleNotFoundError:
    inside_blender: bool = False


from genericpath import exists
from pathlib import Path

if not inside_blender:

    def bake_maps(
        opacity_path: Path,
        albedo_path: Path,
        translucency_path: Path,
        normal_path: Path,
        roughness_path: Path,
        height_path: Path,
        ambient_occlusion_path: Path,
        opacity_repacked_path: Path,
        bake_abc_path: Path,
        texel_density: str,
        asset_name_id: str,
        blender_path: Path,
        output_opacity_path: Path,
        output_albedo_path: Path,
        output_translucency_path: Path,
        output_normal_path: Path,
        output_roughness_path: Path,
        output_height_path: Path,
        output_ao_path: Path,
    ) -> None:
        import subprocess

        subprocess.check_output(
            [
                "sudo",
                str(blender_path),
                "--background",
                "-noaudio",
                "--python",
                "./photometric_stereo_mappings/mappings/bake.py",
                "--",
                str(albedo_path),
                str(ambient_occlusion_path),
                str(opacity_path),
                str(roughness_path),
                str(translucency_path),
                str(normal_path),
                str(height_path),
                str(opacity_repacked_path),
                str(bake_abc_path),
                str(output_albedo_path),
                str(output_ao_path),
                str(output_opacity_path),
                str(output_roughness_path),
                str(output_translucency_path),
                str(output_normal_path),
                str(output_height_path),
                texel_density,
                str(asset_name_id),
            ]
        )


if inside_blender:
    import sys

    argv = sys.argv
    parameters = argv[argv.index("--") + 1 :]

    albedo_path = Path(parameters[0])
    ambient_occlusion_path = Path(parameters[1])
    opacity_path = Path(parameters[2])
    roughness_path = Path(parameters[3])
    translucency_path = Path(parameters[4])
    normal_path = Path(parameters[5])
    height_path = Path(parameters[6])
    opacity_repacked_path = Path(parameters[7])
    bake_abc_path = Path(parameters[8])

    output_albedo_path = Path(parameters[9])
    output_ao_path = Path(parameters[10])
    output_opacity_path = Path(parameters[11])
    output_roughness_path = Path(parameters[12])
    output_translucency_path = Path(parameters[13])
    output_normal_path = Path(parameters[14])
    output_height_path = Path(parameters[15])

    texel_density = int(parameters[16])
    asset_name_id = str(parameters[17])

    import os
    import bpy
    from pathlib import Path
    import time
    import shutil

    def purge_orphans():
        if bpy.app.version >= (3, 0, 0):
            bpy.ops.outliner.orphans_purge(
                do_local_ids=True, do_linked_ids=True, do_recursive=True
            )
        else:
            # call purge_orphans() recursively until there are no more orphan data blocks to purge
            result = bpy.ops.outliner.orphans_purge()
            if result.pop() != "CANCELLED":
                purge_orphans()

    def create_image_node(
        image: bpy.types.Image, nodes: bpy.types.Nodes
    ) -> bpy.types.Node:
        image_node = nodes.new("ShaderNodeTexImage")
        image_node.image = image
        image_node.interpolation = "Closest"
        return image_node

    def create_bake_image(
        opacity: bpy.types.Image,
        name: str,
        nodes: bpy.types.Nodes,
        use_32_bit: bool = False,
    ) -> bpy.types.Image:
        image = bpy.data.images.new(
            name=name,
            width=opacity.size[0],
            height=opacity.size[1],
            float_buffer=use_32_bit,
        )
        image_node = nodes.new("ShaderNodeTexImage")
        image_node.image = image
        image_node.interpolation = "Closest"
        image_node.select = True
        nodes.active = image_node
        return image

    def bake_single_map(
        opacity: bpy.types.Image,
        high_mat: bpy.types.Material,
        mat_output_node: bpy.types.Node,
        image_texture_node: bpy.types.Node,
        name: str,
        nodes: bpy.types.Nodes,
        texture_node: bpy.types.Node,
        output_path: str,
        bit: str,
        colorspace: str,
        display_device: str = "sRGB",
        use_32_bit: bool = False,
    ):
        bpy.context.scene.display_settings.display_device = display_device
        high_mat.node_tree.links.new(
            mat_output_node.inputs[0], image_texture_node.outputs[0]
        )
        bake_image = create_bake_image(opacity, name, nodes, use_32_bit)
        bpy.context.scene.cycles.bake_type = "EMIT"
        bpy.ops.object.bake(type="EMIT")
        texture_node.image = bake_image
        bpy.context.scene.render.filepath = output_path
        bpy.context.scene.render.image_settings.color_mode = colorspace
        bpy.context.scene.render.image_settings.color_depth = bit
        bpy.ops.render.render(write_still=True)

    def blender_bake(
        input_albedo_path: str,
        input_ao_path: str,
        input_opacity_path: str,
        input_roughness_path: str,
        input_translucency_path: str,
        input_normal_path: str,
        input_height_path: str,
        input_opacity_repacked_path: str,
        input_alembic_path: str,
        output_albedo_path: str,
        output_ao_path: str,
        output_opacity_path: str,
        output_roughness_path: str,
        output_translucency_path: str,
        output_normal_path: str,
        output_height_path: str,
        texel_density: int,
        asset_name_id: str,
    ):
        start_time = time.time()

        # Create temp padding folder
        output_folder = Path(output_albedo_path).parent
        padding_image_folder = Path(output_folder, "tmp")
        padding_image_folder.mkdir(exist_ok=True)

        # Check already created maps
        skip_albedo = Path(output_albedo_path).exists()
        skip_ao = Path(output_ao_path).exists()
        skip_opacity = Path(output_opacity_path).exists()
        skip_roughness = Path(output_roughness_path).exists()
        skip_translucency = Path(output_translucency_path).exists()
        skip_normal = Path(output_normal_path).exists()
        skip_height = Path(output_height_path).exists()

        # Remove all objects
        for ob in bpy.data.objects:
            bpy.data.objects.remove(ob)

        # Import alembic file
        bpy.ops.wm.alembic_import(filepath=input_alembic_path)
        high_obj = bpy.data.objects["high"]
        low_obj = bpy.data.objects["low"]

        # Import all textures
        albedo = bpy.data.images.load(filepath=input_albedo_path)
        ao = bpy.data.images.load(filepath=input_ao_path)
        opacity = bpy.data.images.load(filepath=input_opacity_path)
        roughness = bpy.data.images.load(filepath=input_roughness_path)
        translucency = bpy.data.images.load(filepath=input_translucency_path)
        normal = bpy.data.images.load(filepath=input_normal_path)
        height = bpy.data.images.load(filepath=input_height_path)
        opacity_repacked = bpy.data.images.load(filepath=input_opacity_repacked_path)
        white_image = bpy.data.images.new(
            name="White", width=albedo.size[0], height=albedo.size[1]
        )
        white_image.generated_color = (1, 1, 1, 1)

        albedo.use_fake_user = True
        ao.use_fake_user = True
        opacity.use_fake_user = True
        roughness.use_fake_user = True
        translucency.use_fake_user = True
        normal.use_fake_user = True
        height.use_fake_user = True
        opacity_repacked.use_fake_user = True
        white_image.use_fake_user = True

        print("Imported all assets..")

        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.cycles.samples = 1
        bpy.context.scene.cycles.use_denoising = False
        bpy.context.scene.cycles.tile_size = opacity_repacked.size[0]
        bpy.context.scene.render.bake.margin = 0
        bpy.context.scene.view_settings.view_transform = "Standard"
        bpy.data.scenes["Scene"].render.dither_intensity = 0
        print("Render settings setup done...")

        # Add padding to textures using compositor
        bpy.context.scene.use_nodes = True
        comp_nodes = bpy.context.scene.node_tree.nodes
        comp_nodes.clear()

        texture_node = comp_nodes.new(type="CompositorNodeImage")
        white_comp_node = comp_nodes.new(type="CompositorNodeImage")
        opacity_comp_node = comp_nodes.new(type="CompositorNodeImage")
        opacity_comp_node.image = opacity
        set_alpha_node = comp_nodes.new(type="CompositorNodeSetAlpha")
        red_levels_node = comp_nodes.new(type="CompositorNodeLevels")
        red_levels_node.channel = "RED"
        green_levels_node = comp_nodes.new(type="CompositorNodeLevels")
        green_levels_node.channel = "GREEN"
        blue_levels_node = comp_nodes.new(type="CompositorNodeLevels")
        blue_levels_node.channel = "BLUE"
        combRGBA_node = comp_nodes.new(type="CompositorNodeCombRGBA")
        new_alpha_over_node = comp_nodes.new(type="CompositorNodeAlphaOver")
        inpaint_node = comp_nodes.new(type="CompositorNodeInpaint")
        inpaint_node.distance = 16
        composite_node = comp_nodes.new(type="CompositorNodeComposite")
        dilate_erode_node = comp_nodes.new(type="CompositorNodeDilateErode")
        dilate_erode_node.distance = -1
        floor_node = comp_nodes.new("CompositorNodeMath")
        floor_node.operation = "FLOOR"
        alpha_over_node = comp_nodes.new(type="CompositorNodeAlphaOver")
        mixrgb_node = comp_nodes.new("CompositorNodeMixRGB")
        mixrgb_node.use_clamp = True

        # Add padding
        print("Adding padding to textures")
        bpy.context.scene.render.resolution_x = opacity.size[0]
        bpy.context.scene.render.resolution_y = opacity.size[1]

        tree = bpy.context.scene.node_tree
        tree.links.new(set_alpha_node.inputs[0], texture_node.outputs[0])
        tree.links.new(set_alpha_node.inputs[1], floor_node.outputs[0])
        tree.links.new(floor_node.inputs[0], dilate_erode_node.outputs[0])
        tree.links.new(dilate_erode_node.inputs[0], opacity_comp_node.outputs[0])
        tree.links.new(inpaint_node.inputs[0], set_alpha_node.outputs[0])
        tree.links.new(alpha_over_node.inputs[0], opacity_comp_node.outputs[0])
        tree.links.new(alpha_over_node.inputs[1], inpaint_node.outputs[0])
        tree.links.new(alpha_over_node.inputs[2], texture_node.outputs[0])
        tree.links.new(composite_node.inputs[0], alpha_over_node.outputs[0])

        # Albedo padding
        if not skip_albedo:
            print("Padding albedo map...")
            dilate_erode_node.mute = True
            texture_node.image = albedo
            path = Path(padding_image_folder, albedo.name + "_padded.png").as_posix()
            bpy.context.scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            albedo = bpy.data.images.load(filepath=path)
        else:
            print("Skipped padding Albedo")

        # Translucency padding
        if not skip_translucency:
            print("Padding translucency map...")
            texture_node.image = translucency
            path = Path(
                padding_image_folder, translucency.name + "_padded.png"
            ).as_posix()
            bpy.context.scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            translucency = bpy.data.images.load(filepath=path)
        else:
            print("Skipped padding Translucency")

        # AO padding
        if not skip_ao:
            print("Padding ao map...")
            dilate_erode_node.mute = False
            texture_node.image = ao
            path = Path(padding_image_folder, ao.name + "_padded.png").as_posix()
            bpy.context.scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            ao = bpy.data.images.load(filepath=path)
        else:
            print("Skipped padding AO")

        # Roughness padding
        if not skip_roughness:
            print("Padding roughness map...")
            texture_node.image = roughness
            path = Path(padding_image_folder, roughness.name + "_padded.png").as_posix()
            bpy.context.scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            roughness = bpy.data.images.load(filepath=path)
        else:
            print("Skipped padding Roughness")

        # Set ColorSpace
        ao.colorspace_settings.name = "Non-Color"
        opacity.colorspace_settings.name = "Non-Color"
        roughness.colorspace_settings.name = "Non-Color"
        normal.colorspace_settings.name = "Non-Color"
        height.colorspace_settings.name = "Non-Color"
        opacity_repacked.colorspace_settings.name = "Non-Color"

        # Create material for highpoly and lowpoly
        high_mat = bpy.data.materials.new(name="high_mat")
        high_mat.use_nodes = True
        high_mat.cycles.displacement_method = "DISPLACEMENT"

        low_mat = bpy.data.materials.new(name="low_mat")
        low_mat.use_nodes = True
        low_mat.cycles.displacement_method = "DISPLACEMENT"

        print("Created materials...")

        # Assign mat to high obj and low obj
        high_obj.data.materials.append(high_mat)
        low_obj.data.materials.append(low_mat)

        nodes = high_mat.node_tree.nodes
        nodes.clear()
        low_mat.node_tree.nodes.clear()

        # Create material nodes
        material_output_node = nodes.new("ShaderNodeOutputMaterial")
        principled_node = nodes.new("ShaderNodeBsdfPrincipled")
        albedo_node = create_image_node(albedo, nodes)
        white_node = create_image_node(white_image, nodes)
        ao_node = create_image_node(ao, nodes)
        opacity_node = create_image_node(opacity, nodes)
        roughness_node = create_image_node(roughness, nodes)
        translucency_node = create_image_node(translucency, nodes)
        normal_node = create_image_node(normal, nodes)
        normal_map_node = nodes.new("ShaderNodeNormalMap")
        height_node = create_image_node(height, nodes)
        print("Created node tree...")

        high_obj.select_set(True)
        low_obj.select_set(True)
        bpy.context.view_layer.objects.active = low_obj

        bpy.context.scene.render.bake.use_pass_direct = False
        bpy.context.scene.render.bake.use_pass_indirect = False
        bpy.context.scene.render.bake.use_selected_to_active = True

        # Bake opacity map
        high_mat.node_tree.links.new(
            material_output_node.inputs[0], opacity_node.outputs[0]
        )
        opacity_bake_image = create_bake_image(
            opacity_repacked, f"{asset_name_id}_Opacity", low_mat.node_tree.nodes
        )
        bpy.context.scene.cycles.bake_type = "EMIT"
        print("Baking opacity map started")

        bpy.ops.object.bake(type="EMIT")

        # Disconnect padding node tree
        tree.links.new(composite_node.inputs[0], alpha_over_node.outputs[0])
        # Create opacity blend node tree
        opacity_new_node = comp_nodes.new(type="CompositorNodeImage")
        opacity_new_node.image = opacity_bake_image
        opacity_repacked_node = comp_nodes.new(type="CompositorNodeImage")
        opacity_repacked_node.image = opacity_repacked
        math_node = comp_nodes.new(type="CompositorNodeMath")
        math_node.operation = "MAXIMUM"
        floor2_node = comp_nodes.new(type="CompositorNodeMath")
        floor2_node.operation = "FLOOR"
        tree.links.new(math_node.inputs[0], opacity_new_node.outputs[0])
        tree.links.new(math_node.inputs[1], opacity_repacked_node.outputs[0])
        tree.links.new(floor2_node.inputs[0], math_node.outputs[0])
        tree.links.new(composite_node.inputs[0], floor2_node.outputs[0])
        bpy.context.scene.render.filepath = output_opacity_path
        bpy.context.scene.render.resolution_x = opacity_bake_image.size[0]
        bpy.context.scene.render.resolution_y = opacity_bake_image.size[1]
        bpy.context.scene.render.image_settings.color_mode = "BW"
        bpy.context.scene.render.image_settings.color_depth = "8"

        bpy.ops.render.render(write_still=True)

        # reconnect padding node tree
        tree.links.new(alpha_over_node.inputs[1], inpaint_node.outputs[0])
        tree.links.new(composite_node.inputs[0], math_node.outputs[0])
        tree.links.new(composite_node.inputs[0], alpha_over_node.outputs[0])
        tree.links.new(green_levels_node.inputs[0], set_alpha_node.outputs[0])
        tree.links.new(red_levels_node.inputs[0], set_alpha_node.outputs[0])
        tree.links.new(blue_levels_node.inputs[0], set_alpha_node.outputs[0])
        tree.links.new(combRGBA_node.inputs[0], red_levels_node.outputs[0])
        tree.links.new(combRGBA_node.inputs[1], green_levels_node.outputs[0])
        tree.links.new(combRGBA_node.inputs[2], blue_levels_node.outputs[0])
        tree.links.new(new_alpha_over_node.inputs[1], combRGBA_node.outputs[0])
        tree.links.new(new_alpha_over_node.inputs[2], inpaint_node.outputs[0])
        tree.links.new(alpha_over_node.inputs[1], new_alpha_over_node.outputs[0])

        opacity_comp_node.image = opacity_bake_image

        inpaint_node.distance = int(texel_density / 100)

        # Bake White image
        print("Baking white_image")
        high_mat.node_tree.links.new(
            material_output_node.inputs[0], white_node.outputs[0]
        )
        white_bake_image = create_bake_image(
            opacity_repacked, f"{asset_name_id}_white", low_mat.node_tree.nodes
        )
        bpy.context.scene.cycles.bake_type = "EMIT"
        bpy.ops.object.bake(type="EMIT")
        white_comp_node.image = white_bake_image

        # link baked opacity, backed white image and repacked opacity for padding
        tree.links.new(mixrgb_node.inputs[0], opacity_repacked_node.outputs[0])
        tree.links.new(mixrgb_node.inputs[1], opacity_comp_node.outputs[0])
        tree.links.new(mixrgb_node.inputs[2], white_comp_node.outputs[0])
        tree.links.new(dilate_erode_node.inputs[0], mixrgb_node.outputs[0])
        tree.links.new(alpha_over_node.inputs[0], mixrgb_node.outputs[0])

        # Bake albedo
        if not skip_albedo:
            print("Baking albedo map")
            dilate_erode_node.mute = False
            purge_orphans()

            bake_single_map(
                opacity_repacked,
                high_mat,
                material_output_node,
                albedo_node,
                f"{asset_name_id}_Albedo",
                low_mat.node_tree.nodes,
                texture_node,
                output_albedo_path,
                "8",
                "RGB",
            )
            albedo.use_fake_user = False
            purge_orphans()

        # Bake translucency
        if not skip_translucency:
            print("Baking translucency map")
            bake_single_map(
                opacity_repacked,
                high_mat,
                material_output_node,
                translucency_node,
                f"{asset_name_id}_Translucency",
                low_mat.node_tree.nodes,
                texture_node,
                output_translucency_path,
                "8",
                "RGB",
            )
            translucency.use_fake_user = False
            purge_orphans()

        # Bake AO
        if not skip_ao:
            print("Baking ao map")
            dilate_erode_node.mute = False
            bake_single_map(
                opacity_repacked,
                high_mat,
                material_output_node,
                ao_node,
                f"{asset_name_id}_AO",
                low_mat.node_tree.nodes,
                texture_node,
                output_ao_path,
                "8",
                "BW",
                "None",
            )
            ao.use_fake_user = False
            purge_orphans()

        # Bake roughness
        if not skip_roughness:
            print("Baking roughness map")
            bake_single_map(
                opacity_repacked,
                high_mat,
                material_output_node,
                roughness_node,
                f"{asset_name_id}_Roughness",
                low_mat.node_tree.nodes,
                texture_node,
                output_roughness_path,
                "8",
                "BW",
                "None",
            )
            roughness.use_fake_user = False
            purge_orphans()

        # Bake height
        if not skip_height:
            dilate_erode_node.mute = False
            print("Baking height map")
            bake_single_map(
                opacity_repacked,
                high_mat,
                material_output_node,
                height_node,
                f"{asset_name_id}_Height",
                low_mat.node_tree.nodes,
                texture_node,
                output_height_path,
                "16",
                "BW",
                "None",
                True,
            )
            height.use_fake_user = False
            purge_orphans()

        # Bake normal
        if not skip_normal:
            bpy.context.scene.display_settings.display_device = "sRGB"
            bpy.context.scene.cycles.samples = 5
            high_mat.node_tree.links.new(
                material_output_node.inputs[0], principled_node.outputs[0]
            )
            high_mat.node_tree.links.new(
                principled_node.inputs["Normal"], normal_map_node.outputs[0]
            )
            high_mat.node_tree.links.new(
                normal_map_node.inputs["Color"], normal_node.outputs[0]
            )
            normal_bake_image = create_bake_image(
                opacity_repacked, f"{asset_name_id}_Normal", low_mat.node_tree.nodes
            )
            bpy.context.scene.cycles.bake_type = "NORMAL"
            bpy.context.scene.render.bake.normal_b = "NEG_Z"
            print("Baking normal map started")
            bpy.ops.object.bake(type="NORMAL")
            texture_node.image = normal_bake_image
            bpy.context.scene.render.filepath = output_normal_path
            bpy.context.scene.render.image_settings.color_mode = "RGB"
            bpy.context.scene.render.image_settings.color_depth = "8"
            bpy.ops.render.render(write_still=True)

        try:
            print("Trying to delete temp folder")
            shutil.rmtree(padding_image_folder)
        except:
            print("No temp folder for deletion")
        end_time = time.time()
        time_interval = end_time - start_time
        print("Took ", time_interval / 60)

    blender_bake(
        str(albedo_path),
        str(ambient_occlusion_path),
        str(opacity_path),
        str(roughness_path),
        str(translucency_path),
        str(normal_path),
        str(height_path),
        str(opacity_repacked_path),
        str(bake_abc_path),
        str(output_albedo_path),
        str(output_ao_path),
        str(output_opacity_path),
        str(output_roughness_path),
        str(output_translucency_path),
        str(output_normal_path),
        str(output_height_path),
        int(texel_density),
        asset_name_id,
    )
