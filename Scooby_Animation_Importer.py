import bpy
from bpy.props import StringProperty, CollectionProperty, FloatProperty
from bpy_extras.io_utils import ImportHelper

import struct
from mathutils import Quaternion, Vector, Matrix
from pathlib import Path

# Try to hook into the BFBB ANM importer logic for .anm files
# (Using its load() function as an internal helper, not its operator)
try:
    from io_scene_bfbb_anm import import_bfbb_anm
    HAS_BFBB_ANM = True
except Exception:
    HAS_BFBB_ANM = False


# ----------------------------------------------------------------------
# Low-level SKA parsing helpers (Scooby / BFBB style)
# ----------------------------------------------------------------------

def read_uint32_big_endian(data_bytes: bytes, offset: int) -> int:
    return struct.unpack_from(">I", data_bytes, offset)[0]


def read_uint16_big_endian(data_bytes: bytes, offset: int) -> int:
    return struct.unpack_from(">H", data_bytes, offset)[0]


def read_int16_big_endian(data_bytes: bytes, offset: int) -> int:
    return struct.unpack_from(">h", data_bytes, offset)[0]


def read_float32_big_endian(data_bytes: bytes, offset: int) -> float:
    return struct.unpack_from(">f", data_bytes, offset)[0]


def parse_ska_file(file_path: str):
    """
    Parse a Scooby / BFBB SKA animation file.

    Layout (from iAnimSKB.cpp):

        [Header]              0x1C bytes
        [Keyframes]           KeyCount * 16 bytes
        [TimeValues]          TimeCount * 4 bytes
        [KeyOffsetTable]      (TimeCount - 1) * BoneCount * 2 bytes
    """

    path_object = Path(file_path)
    data_bytes = path_object.read_bytes()
    file_size = len(data_bytes)

    if file_size < 0x1C:
        raise ValueError("File is too small to contain a valid SKA/SKB header.")

    header_size_in_bytes = 0x1C

    magic_value = read_uint32_big_endian(data_bytes, 0)
    flags_value = read_uint32_big_endian(data_bytes, 4)
    bone_count = read_uint16_big_endian(data_bytes, 8)
    time_count = read_uint16_big_endian(data_bytes, 10)
    keyframe_count = read_uint32_big_endian(data_bytes, 12)

    scale_values_xyz = [
        read_float32_big_endian(data_bytes, 16),
        read_float32_big_endian(data_bytes, 20),
        read_float32_big_endian(data_bytes, 24),
    ]

    header_information = {
        "magic": hex(magic_value),
        "flags": flags_value,
        "bone_count": bone_count,
        "time_count": time_count,
        "keyframe_count": keyframe_count,
        "scale_values_xyz": scale_values_xyz,
        "file_size_in_bytes": file_size,
    }

    # ---- Layout offsets ----
    keyframe_struct_size = 16

    keyframes_start_offset = header_size_in_bytes
    keyframes_total_size = keyframe_count * keyframe_struct_size

    time_values_start_offset = keyframes_start_offset + keyframes_total_size
    time_values_total_size = time_count * 4

    offset_time_slot_count = max(time_count - 1, 0)
    key_offsets_start_offset = time_values_start_offset + time_values_total_size
    key_offsets_total_size = offset_time_slot_count * bone_count * 2

    total_expected_size = key_offsets_start_offset + key_offsets_total_size

    if total_expected_size > file_size:
        raise ValueError(
            f"File too small for expected SKA layout. "
            f"Need {total_expected_size} bytes, have {file_size} bytes."
        )

    # ---- Read time values ----
    time_values = []
    for time_index in range(time_count):
        offset = time_values_start_offset + time_index * 4
        time_value = read_float32_big_endian(data_bytes, offset)
        time_values.append(time_value)

    # ---- Read keyframes ----
    keyframes = []
    for keyframe_index in range(keyframe_count):
        base_offset = keyframes_start_offset + keyframe_index * keyframe_struct_size

        raw_time_index = read_uint16_big_endian(data_bytes, base_offset + 0)

        raw_quaternion_values = [
            read_int16_big_endian(data_bytes, base_offset + 2),
            read_int16_big_endian(data_bytes, base_offset + 4),
            read_int16_big_endian(data_bytes, base_offset + 6),
            read_int16_big_endian(data_bytes, base_offset + 8),
        ]

        raw_translation_values = [
            read_int16_big_endian(data_bytes, base_offset + 10),
            read_int16_big_endian(data_bytes, base_offset + 12),
            read_int16_big_endian(data_bytes, base_offset + 14),
        ]

        quaternion_floats = [value / 32767.0 for value in raw_quaternion_values]
        length_squared = sum(component * component for component in quaternion_floats)
        if length_squared > 1e-12:
            length = length_squared ** 0.5
            quaternion_floats = [component / length for component in quaternion_floats]

        scaled_translation = [
            raw_translation_values[0] * scale_values_xyz[0],
            raw_translation_values[1] * scale_values_xyz[1],
            raw_translation_values[2] * scale_values_xyz[2],
        ]

        actual_time_value = None
        if 0 <= raw_time_index < len(time_values):
            actual_time_value = time_values[raw_time_index]

        keyframes.append(
            {
                "index": keyframe_index,
                "time_index": raw_time_index,
                "time": actual_time_value,
                "quaternion": quaternion_floats,
                "translation": scaled_translation,
            }
        )

    # ---- Read key offset grid (time_slot x bone) ----
    key_offset_grid = []
    for time_slot_index in range(offset_time_slot_count):
        row = []
        row_offset = key_offsets_start_offset + time_slot_index * bone_count * 2

        for bone_index in range(bone_count):
            keyframe_index_value = read_uint16_big_endian(
                data_bytes,
                row_offset + bone_index * 2,
            )
            row.append(keyframe_index_value)

        key_offset_grid.append(row)

    # ---- Build per-bone animation tracks ----
    bone_animation_tracks = [[] for _ in range(bone_count)]

    for time_slot_index in range(offset_time_slot_count):
        for bone_index in range(bone_count):
            keyframe_index_value = key_offset_grid[time_slot_index][bone_index]

            if keyframe_index_value == 0xFFFF:
                continue
            if keyframe_index_value >= keyframe_count:
                continue

            keyframe = keyframes[keyframe_index_value]

            bone_animation_tracks[bone_index].append(
                {
                    "frame_time_slot_index": time_slot_index,
                    "absolute_time_value": keyframe["time"],
                    "quaternion": keyframe["quaternion"],
                    "translation": keyframe["translation"],
                }
            )

    for track in bone_animation_tracks:
        track.sort(
            key=lambda entry: (
                float("inf") if entry["absolute_time_value"] is None else entry["absolute_time_value"],
                entry["frame_time_slot_index"],
            )
        )

    return {
        "header": header_information,
        "time_values": time_values,
        "bone_animation_tracks": bone_animation_tracks,
    }


# ----------------------------------------------------------------------
# Armature + animation application (for SKA)
# ----------------------------------------------------------------------

def get_active_armature_object() -> bpy.types.Object:
    active_object = bpy.context.object
    if active_object is None:
        raise RuntimeError(
            "No active object. Please select your armature and run the importer again."
        )
    if active_object.type != "ARMATURE":
        raise RuntimeError(
            f"The active object '{active_object.name}' is not an armature. "
            "Please select the armature to animate, then run the importer again."
        )
    return active_object


def build_bone_index_to_pose_bone_map(armature_object: bpy.types.Object) -> dict:
    pose_bones = armature_object.pose.bones
    armature_data = armature_object.data

    bone_index_to_pose_bone = {}
    used_bone_ids = False

    for pose_bone in pose_bones:
        edit_bone = armature_data.bones.get(pose_bone.name)
        if edit_bone is None:
            continue
        bone_id_property = edit_bone.get("bone_id")
        if isinstance(bone_id_property, int):
            bone_index_to_pose_bone[bone_id_property] = pose_bone
            used_bone_ids = True

    if used_bone_ids:
        print("Using edit bone 'bone_id' properties for SKA bone mapping.")
        return bone_index_to_pose_bone

    print("No 'bone_id' properties found. Falling back to pose bone list order.")
    for fallback_index, pose_bone in enumerate(pose_bones):
        bone_index_to_pose_bone[fallback_index] = pose_bone

    return bone_index_to_pose_bone


def create_or_clear_action(armature_object: bpy.types.Object, action_name: str) -> bpy.types.Action:
    if armature_object.animation_data is None:
        armature_object.animation_data_create()

    existing_action = bpy.data.actions.get(action_name)
    if existing_action is None:
        action = bpy.data.actions.new(action_name)
    else:
        for fcurve in list(existing_action.fcurves):
            existing_action.fcurves.remove(fcurve)
        action = existing_action

    armature_object.animation_data.action = action
    return action


def translation_matrix(translation_vector: Vector) -> Matrix:
    return Matrix.Translation(translation_vector)


def local_to_basis_matrix(local_matrix: Matrix, rest_matrix: Matrix, parent_rest_matrix: Matrix) -> Matrix:
    return rest_matrix.inverted_safe() @ (parent_rest_matrix @ local_matrix)


def apply_ska_to_armature(parsed_ska_data: dict, armature_object: bpy.types.Object, action_name: str):
    header = parsed_ska_data["header"]
    time_values = parsed_ska_data["time_values"]
    bone_animation_tracks = parsed_ska_data["bone_animation_tracks"]

    bone_count = int(header.get("bone_count", len(bone_animation_tracks)))
    time_count = int(header.get("time_count", len(time_values) if time_values else 0))

    bone_index_to_pose_bone = build_bone_index_to_pose_bone_map(armature_object)
    armature_data = armature_object.data

    action = create_or_clear_action(armature_object, action_name)

    bpy.context.view_layer.update()
    for pose_bone in armature_object.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)

    frame_offset = 1
    if time_count > 0:
        bpy.context.scene.frame_start = frame_offset
        bpy.context.scene.frame_end = frame_offset + time_count - 1

    for bone_index, bone_track in enumerate(bone_animation_tracks):
        pose_bone = bone_index_to_pose_bone.get(bone_index)
        if pose_bone is None:
            print(f"Skipping SKA bone index {bone_index}: no matching pose bone found.")
            continue

        edit_bone = armature_data.bones.get(pose_bone.name)
        if edit_bone is None:
            print(f"Skipping bone '{pose_bone.name}': no edit bone data available.")
            continue

        rest_matrix = edit_bone.matrix_local.copy()
        if edit_bone.parent is not None:
            parent_rest_matrix = edit_bone.parent.matrix_local.copy()
            local_rest_rotation = (parent_rest_matrix.inverted_safe() @ rest_matrix).to_quaternion()
        else:
            parent_rest_matrix = Matrix.Identity(4)
            local_rest_rotation = rest_matrix.to_quaternion()

        pose_bone.rotation_mode = "QUATERNION"

        entries_by_time_slot_index = {}
        for track_entry in bone_track:
            time_slot_index = track_entry.get("frame_time_slot_index")
            if time_slot_index is None:
                continue
            entries_by_time_slot_index[int(time_slot_index)] = track_entry

        last_rotation_quaternion = None
        last_translation_vector = None

        local_time_count = time_count
        if local_time_count <= 0 and entries_by_time_slot_index:
            local_time_count = max(entries_by_time_slot_index.keys()) + 1

        for time_slot_index in range(local_time_count):
            frame_number = frame_offset + time_slot_index
            track_entry = entries_by_time_slot_index.get(time_slot_index)

            rotation_quaternion = last_rotation_quaternion
            translation_vector = last_translation_vector

            if track_entry is not None:
                quaternion_list = track_entry.get("quaternion")
                translation_list = track_entry.get("translation")

                if quaternion_list is not None:
                    x_value, y_value, z_value, w_value = quaternion_list
                    game_rotation_quaternion = Quaternion((w_value, x_value, y_value, z_value))
                    rotation_quaternion = local_rest_rotation.rotation_difference(game_rotation_quaternion)
                    last_rotation_quaternion = rotation_quaternion

                if translation_list is not None:
                    game_translation_vector = Vector(translation_list)
                    local_translation_matrix = translation_matrix(game_translation_vector)
                    basis_matrix = local_to_basis_matrix(local_translation_matrix, rest_matrix, parent_rest_matrix)
                    translation_vector = basis_matrix.to_translation()
                    last_translation_vector = translation_vector

            if rotation_quaternion is None and translation_vector is None:
                continue

            if rotation_quaternion is not None:
                pose_bone.rotation_quaternion = rotation_quaternion
                pose_bone.keyframe_insert(
                    data_path="rotation_quaternion",
                    frame=frame_number,
                    group=pose_bone.name,
                )

            if translation_vector is not None:
                pose_bone.location = translation_vector
                pose_bone.keyframe_insert(
                    data_path="location",
                    frame=frame_number,
                    group=pose_bone.name,
                )

    print(f"Scooby Animation Importer: finished applying SKA animation to '{armature_object.name}'.")
    print(f"Action created/updated: '{action_name}'")


# ----------------------------------------------------------------------
# Blender Operator + Menu (multi-file SKA + ANM)
# ----------------------------------------------------------------------

class SCOOBY_OT_import_animations(bpy.types.Operator, ImportHelper):
    """Import Scooby SKA and ANM animations onto the selected armature"""
    bl_idname = "import_scene.scooby_animation"
    bl_label = "Scooby Animation (.ska, .anm)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".ska"
    filter_glob: StringProperty(
        default="*.ska;*.anm",
        options={"HIDDEN"},
        maxlen=255,
    )

    files: CollectionProperty(
        name="File Paths",
        type=bpy.types.OperatorFileListElement,
    )
    directory: StringProperty(
        name="Directory",
        description="Directory of the animation files",
        maxlen=1024,
        subtype="DIR_PATH",
    )

    fps: FloatProperty(
        name="FPS",
        description="Value by which the keyframe time is multiplied (used for both SKA and ANM)",
        default=30.0,
        min=1.0,
        max=240.0,
    )

    def execute(self, context):
        # Collect all selected paths (single or multi)
        animation_paths = []

        if self.files:
            directory_path = Path(self.directory)
            for file_element in self.files:
                animation_paths.append(directory_path / file_element.name)
        else:
            animation_paths.append(Path(self.filepath))

        # Get the armature first
        try:
            armature_object = get_active_armature_object()
        except RuntimeError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        # --------------------------------------------------------------
        # Auto-generate bone_id properties using the DFF operator ONCE
        # --------------------------------------------------------------
        if hasattr(bpy.ops.object, "dff_generate_bone_props"):
            view_layer = context.view_layer

            for scene_object in view_layer.objects:
                scene_object.select_set(False)
            armature_object.select_set(True)
            view_layer.objects.active = armature_object

            previous_mode = armature_object.mode
            try:
                bpy.ops.object.mode_set(mode="EDIT")
                try:
                    bpy.ops.armature.select_all(action="SELECT")
                except Exception:
                    pass
                bpy.ops.object.dff_generate_bone_props()
            except Exception as error:
                self.report({"WARNING"}, f"Could not auto-generate bone properties: {error}")
            finally:
                try:
                    bpy.ops.object.mode_set(mode=previous_mode)
                except Exception:
                    bpy.ops.object.mode_set(mode="OBJECT")
        else:
            self.report(
                {"INFO"},
                "DFF 'Generate Bone Properties' operator not found; "
                "assuming bone_id properties are already set.",
            )

        # --------------------------------------------------------------
        # Import each file, dispatch by extension
        # --------------------------------------------------------------
        imported_ska_actions = []
        imported_anm_count = 0
        errors = []

        for path_object in animation_paths:
            suffix = path_object.suffix.lower()

            # --- SKA handled by our custom logic ---
            if suffix == ".ska":
                try:
                    parsed_ska = parse_ska_file(str(path_object))
                except Exception as error:
                    errors.append(f"{path_object.name}: SKA parse error: {error}")
                    continue

                ska_name = path_object.stem
                action_name = f"{ska_name}_from_ska"

                try:
                    apply_ska_to_armature(parsed_ska, armature_object, action_name)
                    imported_ska_actions.append(action_name)
                except Exception as error:
                    errors.append(f"{path_object.name}: SKA apply error: {error}")
                    continue

            # --- ANM handled by BFBB-style ANM loader logic ---
            elif suffix == ".anm":
                if not HAS_BFBB_ANM:
                    errors.append(
                        f"{path_object.name}: Cannot import .anm because the "
                        "BFBB ANM importer module (io_scene_bfbb_anm.import_bfbb_anm) "
                        "is not available or not enabled."
                    )
                    continue

                try:
                    # Optional: remember existing actions so we can see what's new
                    existing_actions = set(bpy.data.actions)

                    # Call the BFBB ANM loader directly.
                    # Its __init__.py uses: import_bfbb_anm.load(context, file_path, fps)
                    import_bfbb_anm.load(context, str(path_object), 30.0)

                    # Count how many new actions appeared (usually 1 per file)
                    new_actions = [act for act in bpy.data.actions if act not in existing_actions]
                    if new_actions:
                        imported_anm_count += len(new_actions)
                    else:
                        # Fallback: assume at least one animation imported if no error was raised
                        imported_anm_count += 1

                    # Try to rename the active action so it matches the filename nicely
                    if armature_object.animation_data and armature_object.animation_data.action:
                        current_action = armature_object.animation_data.action
                        if not current_action.name.lower().startswith(path_object.stem.lower()):
                            current_action.name = f"{path_object.stem}_anm"

                except Exception as error:
                    errors.append(f"{path_object.name}: ANM import error: {error}")
                    continue
        if imported_ska_actions:
            self.report({'INFO'}, f"Imported {len(imported_ska_actions)} SKA action(s).")
        if imported_anm_count:
            self.report({'INFO'}, f"Imported {imported_anm_count} ANM animation(s).")
        if errors:
            for msg in errors:
                self.report({'WARNING'}, msg)

        return {'FINISHED'}




# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------

def menu_func_import(self, context):
    self.layout.operator(
        SCOOBY_OT_import_animations.bl_idname,
        text="Scooby Animation (.ska, .anm)"
    )

classes = (
    SCOOBY_OT_import_animations,
)


def register():
    for class_type in classes:
        bpy.utils.register_class(class_type)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for class_type in reversed(classes):
        bpy.utils.unregister_class(class_type)


if __name__ == "__main__":
    register()

