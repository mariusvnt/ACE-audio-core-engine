"""
prefs_panel_ui.py is responsible for all the UI related elements of the add-on preferences panel.
"""

import bpy # type: ignore
try:
    from bpy.app.handlers import persistent  # type: ignore
except Exception:  # pragma: no cover - Blender provides this at runtime
    def persistent(fn):
        return fn
from . import utils

# --- Data structures ---
class ACE_DependencyItem(bpy.types.PropertyGroup):
    """Stores metadata about a required dependency."""

    library_id: bpy.props.StringProperty(name="Library ID")                      # type: ignore
    display_name: bpy.props.StringProperty(name="Display Name")                  # type: ignore
    description: bpy.props.StringProperty(name="Description")                    # type: ignore
    is_builtin: bpy.props.BoolProperty(name="Built-in", default=False)           # type: ignore


# --- UI Lists ---
class ACE_UL_dependencies(bpy.types.UIList):
    """Displays the list of AudioCore Engine dependencies."""

    bl_description = "List of AudioCore Engine dependencies"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        """Render each dependency with its install state and actions."""
        if not isinstance(item, ACE_DependencyItem):
            return

        is_installed = utils.is_module_installed(item.library_id)
        in_vendor = utils.is_module_in_vendor(item.library_id)
        row = layout.row(align=True)
        row.label(text=item.display_name, icon="CHECKMARK" if is_installed else "ERROR")

        desc = row.row()
        desc.enabled = False
        desc.label(text=item.description)

        if in_vendor:
            # ACE-managed install: allow uninstall
            op = row.operator("audiocore.uninstall_library", text="", icon="TRASH")
            op.library_name = item.library_id
        elif is_installed:
            # Installed externally (system pip, Blender Extensions, etc.): read-only
            row.label(text="External", icon="LINKED")
        else:
            # Not installed: offer install
            op = row.operator("audiocore.install_library", text="", icon="IMPORT")
            op.library_name = item.library_id


# --- Add-on preferences ---
class ACE_preferences(bpy.types.AddonPreferences):
    """Preferences panel for AudioCore Engine."""

    bl_idname = __package__ or __name__.split(".")[0]

    def _update_debug_logging(self, context):
        """Update callback for debug logging toggle."""
        utils.set_debug_logging(self.debug_logging)

    debug_logging: bpy.props.BoolProperty( # type: ignore
        name="Enable Debug Logs",
        description="Print debug logs to the console (errors always print)",
        default=False,
        update=_update_debug_logging
    )
    dependencies: bpy.props.CollectionProperty(type=ACE_DependencyItem) # type: ignore
    dependency_index: bpy.props.IntProperty(default=0) # type: ignore

    def _refresh_dependencies(self, required_libraries):
        """Sync displayed dependencies with the REQUIRED_LIBRARIES list."""
        desired_items = [
            (lib_id, lib_info)
            for lib_id, lib_info in required_libraries.items()
            if not lib_info.get("default", False)
        ]

        current_ids = {item.library_id for item in self.dependencies}
        desired_ids = {lib_id for lib_id, _ in desired_items}
        if current_ids == desired_ids and len(self.dependencies) == len(desired_items):
            return

        self.dependencies.clear()
        for lib_id, lib_info in desired_items:
            item = self.dependencies.add()
            item.library_id = lib_id
            item.display_name = lib_info["name"]
            item.description = lib_info["desc"]
            item.is_builtin = lib_info.get("default", False)

    def draw(self, context):
        layout = self.layout

        layout.label(text="AudioCore Engine relies on external open-source libraries to run. Please install any missing dependencies below.")

        missing_count = sum(1 for item in self.dependencies if not utils.is_module_installed(item.library_id))
        row = layout.row()
        row.enabled = missing_count > 0
        row.operator(
            "audiocore.install_missing_libraries",
            text=f"Install All Missing Libraries ({missing_count})" if missing_count > 0 else "Install All Missing Libraries",
            icon="IMPORT",
        )

        layout.template_list(
            "ACE_UL_dependencies",
            "dependencies",
            self,
            "dependencies",
            self,
            "dependency_index",
            rows=max(3, len(self.dependencies))
        )

        layout.label(text="Blender will freeze during install/uninstall operations. Keep the application running until all operations finish.", icon='INFO')
        layout.label(text="You may need to restart Blender after install/uninstall operations.", icon='INFO')

        layout.separator(type='LINE')

        # Debug logging toggle
        layout.label(text="By default, the add-on only prints errors to the console. Enable extended logs for debugging.")
        layout.prop(self, "debug_logging", text="Extend Logs")


def _get_addon_preferences():
    prefs = getattr(getattr(bpy, "context", None), "preferences", None)
    addons = getattr(prefs, "addons", None) if prefs else None
    if not addons:
        return None
    candidate_keys = [ACE_preferences.bl_idname, __package__, __name__.split(".")[0]]
    for key in candidate_keys:
        if key and key in addons:
            return addons[key].preferences
    return None


def sync_preferences_dependencies():
    try:
        from .vars import REQUIRED_LIBRARIES
    except ImportError:
        return
    pref = _get_addon_preferences()
    if pref and hasattr(pref, "_refresh_dependencies"):
        try:
            pref._refresh_dependencies(REQUIRED_LIBRARIES)
        except Exception:
            pass


@persistent
def _prefs_load_post(_dummy):
    sync_preferences_dependencies()


# --- Registration ---
classes = (
    ACE_DependencyItem,
    ACE_UL_dependencies,
    ACE_preferences,
)

def register():
    """Register dependency UI and preferences for the add-on."""
    utils.log("Registering ui_preferences.py...", 'DEBUG')
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            pass # Already registered
    sync_preferences_dependencies()
    if _prefs_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_prefs_load_post)

def unregister():
    """Unregister dependency UI and preferences for the add-on."""
    utils.log("Unregistering ui_preferences.py...", 'DEBUG')
    if _prefs_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_prefs_load_post)
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, AttributeError):
            pass
