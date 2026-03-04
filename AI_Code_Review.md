# ACE Architectural Code Review Prompt

You are conducting a deep architectural code review of the AudioCore Engine (ACE) codebase.

You must review the implementation against the current refactored architecture, not the legacy single-tree Geometry Nodes model. Be skeptical, evidence-driven, and findings-first. Your job is to identify crash risks, architectural regressions, determinism bugs, scene-compile flaws, Stage C sink-routing issues, and missing tests.

## Primary Objective

Determine whether ACE correctly implements the scene-scoped multi-frontend architecture:
- Stage A discovers all managed scene frontends, parses them independently, and merges them into one `ACEAbstractTree`
- Stage B remains `bpy`-free and still keys off `graph.output_attr_keys`
- Stage C resolves concrete sink bindings and fans results out to proxy attributes and object custom properties on the main thread

Do not grade the code against outdated assumptions such as:
- a single active GN tree being the real compile boundary
- source-node identity being `ace_id == ace_source_id`
- Stage C only writing Geometry proxy attributes

## Current Architecture Baseline

### 1. Scene-Scoped Frontends

ACE frontends are top-level Blender node trees discovered from the scene:
- Geometry frontend: every top-level `GeometryNodeTree` referenced by an object's `NODES` modifier
- Shader frontend: every top-level `ShaderNodeTree` from a material assigned to a scene object

Discovery rules:
- deduplicate by top-level `NodeTree` datablock, not by object usage
- ignore internal `.ACE_*` implementation groups
- a tree counts as ACE-managed if it has `ace_frontend_uid` or contains ACE nodes
- shared materials are one shader frontend with multiple consumer objects
- frontends must be sorted by `frontend_uid` before parse/merge

### 2. Merged IR

The scene IR is `ACEAbstractTree`.

It must include:
- `frontends`
- `consumer_intents`
- frontend metadata on nodes and output links
- source-binding metadata on source nodes

Identity rules:
- `ACEAbstractNode.id` is frontend-scoped and unique per node instance
- `ace_source_id` remains the stable source binding token
- `ACEAbstractNode.source_binding_id` mirrors that stable token in IR
- source placeholder nodes must not rely on `ace_id == ace_source_id`

Preview rule:
- `group_output_links` remain relevant for GN preview auto-taps
- they must carry frontend metadata
- preview auto-taps remain Geometry-only in V1

### 3. Compile Model

The primary Stage A entry point is:
- `compile_scene_frontends(scene, preferred_tree=None)`

Compatibility rule:
- `compile_from_blender_tree(bl_tree, scene)` may remain public, but it is only a wrapper around scene-wide compile

Compile rules:
- each frontend is parsed independently
- one shared reserved-node-ID set is used during parse
- all partial trees merge into one scene `ACEAbstractTree`
- `compile(abstract_tree, scene)` still builds per-source graphs from merged IR
- the scene signature must hash `frontends` and `consumer_intents`, not just nodes/links/output-links

### 4. Stage B Boundary

Stage B remains structurally unchanged:
- worker protocol still keys off `graph.output_attr_keys`
- worker compute remains `bpy`-free
- no frontend-specific Blender logic should leak into workers

### 5. Stage C Sink Dispatch

`ACEProxyManager` is now effectively a sink coordinator.

Runtime rules:
- `graph.output_attr_keys` remains the compute identity list
- `graph.sink_bindings_by_attr` is the concrete Stage C write plan
- each `attr_key` may resolve to one or more concrete sinks

Supported sink kinds:
- `PROXY_ATTRIBUTE`
- `OBJECT_PROPERTY`

Sink rules:
- GN outputs resolve to proxy mesh attributes
- Shader outputs resolve to object custom properties
- both sink kinds must reuse the same deterministic compact naming path
- Shader V1 is object-bound materials only; World and Light shaders are out of scope
- if writes are unsafe during render, Stage C must skip all sink kinds consistently

### 6. Dirty Detection

Dirty-state handling must be scene-aware.

It should react to:
- `GeometryNodeTree` updates
- `ShaderNodeTree` updates
- `Material` updates that affect shader frontends
- relevant `Object` updates that change frontend membership

Loop-prevention rule:
- object property writes must not create perpetual recompiles
- frontend-membership signature gating should prevent self-triggered compile storms

### 7. Layer 3 Constraints

Layer 3 remains isolated:
- time-domain preview only
- no inverse FFT, inverse BFT, overlap-add reconstruction, or phase-vocoder style recovery
- preview auto-taps are still GN-only in V1

## What You Must Verify

### A. Main-Thread Safety
- any `bpy` mutation outside the main thread
- missing `utils.assert_main_thread()` guards in Stage A or Stage C paths
- missing or inconsistent render-write safety checks

### B. Frontend Discovery and Identity
- incorrect frontend discovery scope
- duplicate frontend UID handling that is non-deterministic
- internal `.ACE_*` groups accidentally treated as frontends
- node-ID collisions across merged frontends
- source-node identity regressions where `ace_id` still aliases the stable source token

### C. IR and Compile Correctness
- missing `frontends` or `consumer_intents` from IR
- scene signature not incorporating frontend/consumer semantics
- compile flow still depending on a single GN tree in critical runtime paths
- wrapper compatibility regressions for `compile_from_blender_tree()`

### D. Stage C Fan-Out Correctness
- missing `sink_bindings_by_attr`
- object-property sinks not using object custom properties
- divergent naming schemes between GN and Shader sinks
- render guard skipping one sink kind but not the other
- missing delta suppression or cache usage in sink writers

### E. Boundary Preservation
- `bpy` contamination in Stage B workers
- Layer 2 / Layer 3 contamination
- preview reconstruction using frequency-domain output

### F. Test Coverage

Check that the codebase has regression coverage, or call out missing coverage, for:
- frontend UID persistence and duplicate repair
- source-node identity separation
- merged scene signature stability
- multi-GN compile
- mixed GN + Shader compile
- Stage C object-property writes
- render-guard behavior
- dirty detection for Geometry trees, Shader trees, Material changes, and Object membership changes

## High-Severity Failure Conditions

Treat the following as severe architectural failures:
- any `bpy` access in worker processes
- source placeholder nodes still using the stable source token as their node identity
- scene signature still ignoring `frontends` or `consumer_intents`
- cross-frontend node-ID collisions after merge
- Shader sinks implemented through drivers, textures, or other non-property paths in V1
- Stage C updating Geometry sinks while skipping Shader sinks, or vice versa, under render guard
- object property writes retriggering endless scene recompiles

## Review Instructions

- Findings come first. Do not start with praise or a high-level summary.
- Order findings by severity.
- Use exact file and line references.
- Explain why each issue matters in terms of architecture, runtime correctness, determinism, or safety.
- Distinguish verified facts from inference.
- If no findings are discovered, say that explicitly and then list residual risks or testing gaps.

## Required Output Format

### Findings
- one item per issue
- include severity, impact, evidence, and expected behavior

### Open Questions or Assumptions
- only include unresolved ambiguities that affect confidence

### Residual Risks or Missing Tests
- list any important behavior that appears under-tested

### Verdict
- `architecture aligned`
- `architecture partially aligned`
- `architecture not aligned`

If you provide a score, only score criteria you actually verified in code. Do not invent confidence or estimated points.




#### Needed modifications

In the Engine Status section in the n panel, the "Engine" label and its gear icon should be removed

The idle/active/refreshing witness should be directly in the engine status panel rather than in a box and the "refreshing" witness should be renamed to "Compiling"fix those

Since there can be multiple ACE node trees now, we need a better UX for the source node group management. I think we should make it visible in the add menu. And make multiple suggestions on how we can construct a clear UX in the n panel with this new need for multiple instances management.

Source node group should not appear in the active nodes list

The geometry target panel should be renamed to Active ACE Instances and be a list of the current node trees containing ACE nodes and the objects they are on (and wether they are shader or geometry editor)

The last audio preview placed should be the only one emitting audio (to solve multi-audio preview conflicts)

#

1. Engine Status Panel Cleanup
We need to streamline the top of the N-panel to remove clutter and make the engine states immediate and clear.

Implementation Logic:

Remove "Engine" / Gear: Strip the layout.label(text="Engine", icon='PREFERENCES') or similar from the panel's draw function.

Flatten Witness & Rename: Remove the box() container around the status. Update the enum or label strings so that REFRESHING is explicitly rendered as Compiling.

Example Snippet (ui.py):

Python
def draw(self, context):
    layout = self.layout
    scene = context.scene
    ace_props = scene.ace_props # Or your relevant property group
    
    # Flattened status, no box, no gear icon
    row = layout.row(align=True)
    
    if ace_props.engine_status == 'IDLE':
        row.label(text="Idle", icon='PAUSE')
    elif ace_props.engine_status == 'ACTIVE':
        row.label(text="Active", icon='PLAY')
    elif ace_props.engine_status == 'COMPILING': # Renamed from REFRESHING
        row.label(text="Compiling...", icon='FILE_REFRESH')
2. "Active ACE Instances" Panel (Formerly Geometry Target)
Since ACE now discovers all managed top-level trees (Geometry and Shader), this panel must iterate through the newly proposed ACEFrontendRef data (or its Blender property equivalent) and display a clear ledger of where ACE is operating.

Implementation Logic:

Rename the panel class label to "Active ACE Instances".

Iterate through the scene's managed frontends.

Display the Tree Name, the assigned Object, and an icon denoting its type.

Example Snippet (ui.py):

Python
def draw(self, context):
    layout = self.layout
    # Assuming frontends are cached or accessible via a scene property collection
    frontends = get_scene_frontends(context.scene) 
    
    if not frontends:
        layout.label(text="No active ACE instances.")
        return
        
    for ref in frontends:
        row = layout.row()
        # Visual differentiation based on frontend kind
        icon = 'NODETREE' if ref.kind == "GEOMETRY_NODES" else 'SHADING_RENDERED'
        
        row.label(text=ref.tree_name, icon=icon)
        if ref.owner_keys:
            # Show the primary object owner
            row.label(text=f"on {ref.owner_keys[0]}", icon='OBJECT_DATA')
3. Source Node Visibility & List Filtering
To support multi-instance management, the Source node must be an explicit, user-placeable contract, but it shouldn't clutter the "Active Nodes" execution list since it doesn't compute data itself.

Implementation Logic:

Add Menu: Remove the internal/hidden flag from the Source node in ACE_NODES (vars.py). Ensure it registers with the standard Add Menu (nodeitems_utils or the 5.0 equivalent).

Active Nodes List: In the UIList or drawing loop that renders the "Active Nodes", add a strict filter:
if node.ace_category == 'source_mngmt': continue

4. Resolving Multi-Audio Preview Conflicts (Layer 3)
If a user places multiple Preview nodes across different frontends (or duplicates them), Layer 3 will attempt to mix or overlap them, causing chaos. We need a "last placed wins" or "last modified wins" resolution.

Implementation Logic:

When Stage A parses the scene and merges ACEAbstractTree, it should track a creation_timestamp or rely on the order of discovery.

In _compile_preview_graph() (Layer 3 setup), scan all active preview links.

Sort them by their timestamp or ID, isolate the single most recently added PreviewSinkTask, and aggressively mute or prune the routing for all others.

#

System Context & Role:
You are an expert Blender Python Addon Developer working on the AudioCore Engine (ACE). You must adhere strictly to the ACE Architecture Brief: bpy access is allowed in UI/main-thread code, but no signal processing or frequency-domain logic is permitted in the UI layer.

Task:
Implement a new N-Panel in the 3D Viewport titled "Audio Sources" (ACE_PT_AudioSources). This panel will display a hierarchical "Source-to-Instance Tree View" mapping each active Audio Source to the specific frontends (Geometry Nodes or Shader materials) that are consuming it.

Architectural Data Availability:
Assume that after Stage A compilation, the scene's ACEAbstractTree is available and contains:

nodes: Dict of ACEAbstractNode (where source_mngmt category nodes represent our audio sources).

frontends: Dict of ACEFrontendRef (metadata about discovered Geometry and Shader trees).

consumer_intents: List of ACEConsumerIntent (declarations of which frontend consumes which output, containing source_token / source_binding_id).

Implementation Requirements:

The Panel Class (ACE_PT_AudioSources):

Register it in the 3D Viewport (VIEW_3D), UI region (UI), in the ACE category.

Set the label to "Audio Sources" and use a sound-related icon (e.g., 'OUTLINER_OB_SPEAKER').

The Draw Loop Logic (Fake Hierarchy):

Blender's native UIList does not support nested trees easily. Instead, use a custom draw method that iterates through the data and uses indentation to create a visual tree.

Step 1: Fetch the current scene's compiled IR (abstract_tree). If none exists, display a label: "Engine offline or no sources found."

Step 2: Isolate all unique Audio Sources (nodes where ace_category == 'source_mngmt').

Step 3: For each Source, draw a top-level box or row. Show the Source's custom name or file binding and a volume/mute toggle if available.

Step 4: Query the consumer_intents (or node lineage) to find all unique frontend_uids that are routed to this specific source's source_binding_id.

Step 5: For each consuming frontend, draw an indented row underneath the source.

Use layout.row().separator(factor=2.0) or split to create visual indentation.

Display the frontend's name (frontend_ref.tree_name).

If frontend_ref.kind == "GEOMETRY_NODES", use the 'NODETREE' icon.

If frontend_ref.kind == "SHADER_NODES", use the 'SHADING_RENDERED' icon.

Append the owner object name if available (frontend_ref.owner_keys[0]).

Strict Constraints:

Do not execute any DSP or modify the audio graph here. This is a read-only visual ledger of Layer 1 intent.

Handle cases where a source exists but has no consuming frontends yet (display an indented label: "No connected instances").
