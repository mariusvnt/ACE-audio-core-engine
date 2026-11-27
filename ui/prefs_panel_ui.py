"""prefs_panel_ui.py: the add-on preferences (the extended-logs option)."""

import bpy # type: ignore
from .. import utils

# The add-on's root package: "bl_ext.<repository>.audio_core_engine" as an extension.
ADDON_PACKAGE = (__package__ or __name__).rsplit(".", 1)[0]


class ACE_preferences(bpy.types.AddonPreferences):
    """Preferences panel for Audio Core Engine."""

    bl_idname = ADDON_PACKAGE

    def _update_debug_logging(self, context):
        utils.set_debug_logging(self.debug_logging)

    debug_logging: bpy.props.BoolProperty( # type: ignore
        name="Enable Debug Logs",
        description="Print debug logs to the console (errors always print)",
        default=False,
        update=_update_debug_logging
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="By default, the add-on only prints errors to the console. Enable extended logs for debugging.")
        layout.prop(self, "debug_logging", text="Extend Logs")


def _get_addon_preferences():
    prefs = getattr(getattr(bpy, "context", None), "preferences", None)
    addons = getattr(prefs, "addons", None) if prefs else None
    if not addons or ADDON_PACKAGE not in addons:
        return None
    return addons[ADDON_PACKAGE].preferences


classes = (
    ACE_preferences,
)

def register():
    """Register the add-on preferences."""
    utils.log("Registering ui_preferences.py...", 'DEBUG')
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            pass # Already registered

def unregister():
    """Unregister the add-on preferences."""
    utils.log("Unregistering ui_preferences.py...", 'DEBUG')
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, AttributeError):
            pass
