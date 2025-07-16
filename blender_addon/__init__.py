bl_info = {
    "name": "2D to 3D Face Reconstruction",
    "blender": (3, 6, 1),
    "category": "Object",
    "author": "Soroush Zendedel",
    "version": (1, 0, 0),
    "description": "Convert 2D face images to 3D face meshes using template meshes.",
    "support": "COMMUNITY",
    "wiki_url": "",
    "tracker_url": "",
}

import bpy
import subprocess
import sys
import os
import cv2
import dlib
import numpy as np
from mathutils import Matrix, Vector


class InstallModulesOperator(bpy.types.Operator):
    """Install required Python modules using Blender's bundled pip"""
    bl_idname = "object.install_modules"
    bl_label = "Install Python Modules"

    def execute(self, context):
        python_exe = sys.executable
        required_modules = ['opencv-python', 'dlib', 'numpy']
        for module in required_modules:
            subprocess.check_call([python_exe, '-m', 'pip', 'install', module])
        self.report({'INFO'}, "Modules installed successfully!")
        return {'FINISHED'}


class FaceReconstructionPanel(bpy.types.Panel):
    """UI Panel for the add-on"""
    bl_label = "2D to 3D Face Reconstruction"
    bl_idname = "OBJECT_PT_face_reconstruction"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tools'

    def draw(self, context):
        layout = self.layout
        layout.operator("object.install_modules", text="Install Required Modules")
        layout.operator("object.create_3d_face_mesh", text="Create 3D Face Mesh")


class Create3DFaceMeshOperator(bpy.types.Operator):
    """Generate a 3D face mesh from an image"""
    bl_idname = "object.create_3d_face_mesh"
    bl_label = "Create 3D Face Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    image_path: bpy.props.StringProperty(
        name="Image Path",
        subtype='FILE_PATH'
    )
    landmark_model_path: bpy.props.StringProperty(
        name="Landmark Model Path",
        subtype='FILE_PATH'
    )
    template_mesh_path: bpy.props.StringProperty(
        name="Template Mesh Path",
        subtype='FILE_PATH'
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        image_path = bpy.path.abspath(self.image_path)
        landmark_model_path = bpy.path.abspath(self.landmark_model_path)
        template_mesh_path = bpy.path.abspath(self.template_mesh_path)

        if not os.path.exists(image_path):
            self.report({'ERROR'}, "Invalid image path")
            return {'CANCELLED'}
        if not os.path.exists(landmark_model_path):
            self.report({'ERROR'}, "Invalid landmark model path")
            return {'CANCELLED'}
        if not os.path.exists(template_mesh_path):
            self.report({'ERROR'}, "Invalid template mesh path")
            return {'CANCELLED'}

        # Step 1: Load the image
        image = cv2.imread(image_path)
        if image is None:
            self.report({'ERROR'}, "Failed to load image")
            return {'CANCELLED'}

        # Step 2: Detect facial landmarks
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(landmark_model_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        if not faces:
            self.report({'ERROR'}, "No faces found")
            return {'CANCELLED'}
        shape = predictor(gray, faces[0])
        landmarks = np.array([(pt.x, pt.y) for pt in shape.parts()], dtype=np.float32)

        # Use 6 key landmarks for pose estimation
        image_points = np.array([
            landmarks[30],  # Nose tip
            landmarks[8],   # Chin
            landmarks[36],  # Left eye left corner
            landmarks[45],  # Right eye right corner
            landmarks[48],  # Left Mouth corner
            landmarks[54],  # Right mouth corner
        ], dtype=np.float32)

        # Standard 3D model points of corresponding features
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float32)

        height, width = image.shape[:2]
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            self.report({'ERROR'}, "Pose estimation failed")
            return {'CANCELLED'}

        # Step 3: Import template mesh
        bpy.ops.import_scene.obj(filepath=template_mesh_path)
        obj = context.selected_objects[0]
        context.view_layer.objects.active = obj

        rot_matrix, _ = cv2.Rodrigues(rotation_vector)
        rot_matrix = Matrix(rot_matrix).to_4x4()
        obj.matrix_world = rot_matrix
        obj.location = Vector(translation_vector.flatten()) * 0.001  # scale translation

        # Step 4: Apply the image as texture
        tex_image = bpy.data.images.load(image_path)
        material = bpy.data.materials.new(name="FaceMaterial")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        bsdf = nodes.get('Principled BSDF')
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.image = tex_image
        links.new(bsdf.inputs['Base Color'], tex_node.outputs['Color'])
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

        self.report({'INFO'}, "3D Face Mesh created successfully!")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(InstallModulesOperator)
    bpy.utils.register_class(FaceReconstructionPanel)
    bpy.utils.register_class(Create3DFaceMeshOperator)


def unregister():
    bpy.utils.unregister_class(InstallModulesOperator)
    bpy.utils.unregister_class(FaceReconstructionPanel)
    bpy.utils.unregister_class(Create3DFaceMeshOperator)


if __name__ == "__main__":
    register()
