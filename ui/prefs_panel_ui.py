"""prefs_panel_ui.py: the add-on preferences, the status of the bundled libraries and their switches."""

import bpy # type: ignore
from .. import deps, utils

# The add-on's root package: "bl_ext.<repository>.audio_core_engine" as an extension.
ADDON_PACKAGE = (__package__ or __name__).rsplit(".", 1)[0]

_STATE_ICONS = {"ok": "CHECKMARK", "off": "CANCEL", "missing": "ERROR"}


def _switch_property(key: str, name: str, description: str):
    """A checkbox that reads and writes the deps settings file instead of the preferences datablock."""
    return bpy.props.BoolProperty(
        name=name,
        description=description,
        get=lambda _self: deps.setting(key),
        set=lambda _self, value: deps.set_setting(key, value),
    )


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
    use_numba: _switch_property(  # type: ignore
        "use_numba",
        "Numba",
        "Compile filters and envelopes with Numba. Turn off if it fails on this machine; applies after restarting Blender",
    )
    use_sounddevice: _switch_property(  # type: ignore
        "use_sounddevice",
        "Live Input and Preview Playback",
        "Use SoundDevice for live audio input and the preview monitor; applies the next time a live source or the preview opens",
    )

    def draw(self, context):
        layout = self.layout

        layout.label(text="Libraries bundled with the extension")
        box = layout.box()
        for row_info in deps.report():
            row = box.row()
            row.label(text=row_info["name"], icon=_STATE_ICONS.get(row_info["state"], "QUESTION"))
            desc = row.row()
            desc.enabled = False
            desc.label(text=row_info["desc"])
            status = row_info["version"] if row_info["state"] == "ok" else row_info["detail"]
            row.label(text=status)

        col = layout.column(heading="Use")
        col.prop(self, "use_numba")
        if deps.setting("use_numba") != deps.numba_enabled() and deps.available("numba"):
            col.label(text="Restart Blender to apply the Numba change", icon="INFO")
        col.prop(self, "use_sounddevice")

        layout.separator(type='LINE')
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
