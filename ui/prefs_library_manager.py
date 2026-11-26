"""prefs_library_manager.py contains the install/un-install operators for external libraries that are called in the add-on preferences panel."""

import bpy # type: ignore
import subprocess
import os
import re
import shutil
from .. import utils


def _dependency_target_dir(create: bool = True) -> str:
    return utils.ensure_dependency_vendor_path(create=create)


def _install_library_with_pip(python_exe, lib_name, target_dir):
    install_cmd = [
        python_exe,
        "-m", "pip", "install",
        "--target", target_dir,
        "--upgrade",
        "--prefer-binary",
        "--no-cache-dir",
        "--disable-pip-version-check",
        lib_name
    ]
    process = subprocess.Popen(
        install_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    stdout_data, stderr_data = process.communicate()
    if process.returncode != 0:
        utils.log(f"Installation failed with return code {process.returncode}", type='ERROR')
        utils.log(f"--- PIP STDOUT ({lib_name}) ---", type='ERROR')
        utils.log(stdout_data, type='ERROR')
        utils.log(f"--- PIP STDERR ({lib_name}) ---", type='ERROR')
        utils.log(stderr_data, type='ERROR')
        return False
    return True


def _normalized_token(value: str) -> str:
    return re.sub(r"[-_.]+", "", str(value or "").strip().lower())


def _remove_library_from_target(lib_name: str, target_dir: str) -> bool:
    if not target_dir or not os.path.isdir(target_dir):
        return False
    token = _normalized_token(lib_name)
    if not token:
        return False

    removed = False
    try:
        entries = list(os.listdir(target_dir))
    except Exception as exc:
        utils.note_swallowed("prefs_library_manager._remove_library_from_target", exc)
        return False

    for entry in entries:
        raw_name = str(entry or "")
        if not raw_name:
            continue
        base = raw_name
        if base.endswith(".dist-info"):
            base = base[: -len(".dist-info")]
        elif base.endswith(".egg-info"):
            base = base[: -len(".egg-info")]
        elif base.endswith(".data"):
            base = base[: -len(".data")]
        base_token = _normalized_token(base)
        if not base_token:
            continue
        if not (base_token.startswith(token) or token.startswith(base_token)):
            continue

        path = os.path.join(target_dir, raw_name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=False)
            else:
                os.remove(path)
            removed = True
        except Exception as e:
            utils.log(f"Failed to remove '{path}': {e}", type='ERROR')
    return removed


class ACE_OT_install_library(bpy.types.Operator):
    """Install a required Python library via Blender's pip."""

    bl_idname = "audiocore.install_library"
    bl_label = "Install Library"
    bl_description = "Install a Python library using Blender's pip"
    bl_options = {'REGISTER', 'INTERNAL'}

    library_name: bpy.props.StringProperty(name="Library Name") # type: ignore

    def execute(self, context):
        """Run pip install for the specified library."""
        python_exe = utils.get_python_executable()
        lib_name = self.library_name
        target_dir = _dependency_target_dir(create=True)

        utils.log(f"Attempting installation of {lib_name}...")

        try:
            if not _install_library_with_pip(python_exe, lib_name, target_dir):
                self.report({'ERROR'}, f"Failed to install {lib_name}. Check the system console.")
                return {'CANCELLED'}

            utils.log(f"Successfully installed {lib_name}")

            self.report({'INFO'}, f"{lib_name} installed. Refreshing UI...")
            utils.refresh_import_caches()
            with utils.logged_errors("prefs_library_manager.ACE_OT_install_library.execute"):
                from .prefs_panel_ui import sync_preferences_dependencies
                sync_preferences_dependencies()
            return {'FINISHED'}

        except Exception as e:
            utils.log(f"Critical error during install: {e}", type='ERROR')
            self.report({'ERROR'}, f"Failed to install {lib_name}. Check the system console.")
            return {'CANCELLED'}


class ACE_OT_install_missing_libraries(bpy.types.Operator):
    """Install every missing optional dependency in one pass."""

    bl_idname = "audiocore.install_missing_libraries"
    bl_label = "Install Missing Libraries"
    bl_description = "Install all missing Audio Core dependencies using Blender's pip"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        python_exe = utils.get_python_executable()
        target_dir = _dependency_target_dir(create=True)
        try:
            from ..vars import REQUIRED_LIBRARIES
        except Exception:
            self.report({'ERROR'}, "Unable to resolve dependency registry.")
            return {'CANCELLED'}

        missing = []
        for lib_id, lib_info in REQUIRED_LIBRARIES.items():
            if bool(lib_info.get("default", False)):
                continue
            if not utils.is_module_installed(lib_id):
                missing.append(lib_id)

        if not missing:
            self.report({'INFO'}, "All optional libraries are already installed.")
            return {'FINISHED'}

        installed = []
        failed = []
        for lib_name in missing:
            utils.log(f"Attempting installation of {lib_name}...")
            try:
                ok = _install_library_with_pip(python_exe, lib_name, target_dir)
            except Exception as e:
                ok = False
                utils.log(f"Critical error during install of {lib_name}: {e}", type='ERROR')
            if ok:
                installed.append(lib_name)
                utils.log(f"Successfully installed {lib_name}")
            else:
                failed.append(lib_name)

        utils.refresh_import_caches()
        with utils.logged_errors("prefs_library_manager.ACE_OT_install_missing_libraries.execute"):
            from .prefs_panel_ui import sync_preferences_dependencies
            sync_preferences_dependencies()

        if failed:
            self.report(
                {'WARNING'},
                f"Installed {len(installed)}/{len(missing)} libraries. Failed: {', '.join(failed)}."
            )
            return {'FINISHED'}

        self.report({'INFO'}, f"Installed {len(installed)} missing libraries.")
        return {'FINISHED'}


class ACE_OT_uninstall_library(bpy.types.Operator):
    """Uninstall a Python library."""

    bl_idname = "audiocore.uninstall_library"
    bl_label = "Uninstall Library"
    bl_options = {'REGISTER', 'INTERNAL'}

    library_name: bpy.props.StringProperty(name="Library Name") # type: ignore

    def execute(self, context):
        """Remove a Python library from Audio Core's local vendor directory."""
        lib_name = self.library_name
        target_dir = _dependency_target_dir(create=True)

        try:
            removed = _remove_library_from_target(lib_name, target_dir)
            if not removed:
                # Distinguish between "installed elsewhere" and "not installed at all"
                # so the user knows exactly what to do.
                utils.refresh_import_caches()
                if utils.is_module_installed(lib_name):
                    self.report(
                        {'WARNING'},
                        f"{lib_name} is installed in Blender's Python environment but was "
                        f"not installed via Audio Core's installer — it cannot be managed here. "
                        f"Use 'pip uninstall {lib_name}' in Blender's Python to remove it.",
                    )
                else:
                    self.report(
                        {'WARNING'},
                        f"{lib_name} is not installed. Use the Install button to install it.",
                    )
                return {'CANCELLED'}
            utils.log(f"Successfully removed {lib_name} from {target_dir}")

            self.report({'INFO'}, f"{lib_name} uninstalled. Refreshing UI...")
            # Force import cache refresh so prefs detect removal immediately
            utils.refresh_import_caches()
            with utils.logged_errors("prefs_library_manager.ACE_OT_uninstall_library.execute"):
                from .prefs_panel_ui import sync_preferences_dependencies
                sync_preferences_dependencies()
            return {'FINISHED'}
        except Exception as e:
            utils.log(f"Uninstall failed for {lib_name}: {e}", type='ERROR')
            self.report({'ERROR'}, f"Uninstall failed for {lib_name}")
            return {'CANCELLED'}

classes = (
    ACE_OT_install_library,
    ACE_OT_install_missing_libraries,
    ACE_OT_uninstall_library,
)

def register():
    """Register library management operators."""
    utils.log("Registering library_manager.py...", 'DEBUG')
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            pass # Already registered

def unregister():
    """Unregister library management operators."""
    utils.log("Unregistering library_manager.py...", 'DEBUG')
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, AttributeError):
            pass
