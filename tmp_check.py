import sys
from pathlib import Path

ROOT = Path(r"c:\Users\mariu\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons\AudioCore Engine")
sys.path.append(str(ROOT))

try:
    # Use synthetic package like module_loader if needed, 
    # but let's try direct first.
    import proxy_management
    print(f"Direct import proxy_management version has SONAR_OBJECT_NAME: {hasattr(proxy_management.ACEProxyManager, 'SONAR_OBJECT_NAME')}")
    if hasattr(proxy_management.ACEProxyManager, 'SONAR_OBJECT_NAME'):
        print(f"Value: {proxy_management.ACEProxyManager.SONAR_OBJECT_NAME}")
except Exception as e:
    print(f"Error: {e}")

from tests.module_loader import load_module
ae = load_module("audio_engine")
pm = load_module("proxy_management")
print(f"Module loader proxy_management has SONAR_OBJECT_NAME: {hasattr(pm.ACEProxyManager, 'SONAR_OBJECT_NAME')}")
if hasattr(pm.ACEProxyManager, 'SONAR_OBJECT_NAME'):
    print(f"Value: {pm.ACEProxyManager.SONAR_OBJECT_NAME}")

print(f"audio_engine.ACEProxyManager hasattr SONAR_OBJECT_NAME: {hasattr(ae.ACEProxyManager, 'SONAR_OBJECT_NAME')}")
