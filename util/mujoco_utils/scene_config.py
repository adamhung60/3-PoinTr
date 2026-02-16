"""
Scene configuration utilities for multi-scene support.

Provides scene name detection from XML paths and XML path resolution from scene names.
Scene-specific functionality (filtering, success checking) is now handled
by task automators in util.taskautomators.
"""
import os


def get_xml_path_from_scene_name(scene_name: str) -> str:
    """
    Get XML file path from scene name.
    
    Args:
        scene_name: Scene name (e.g., 'blockstack', 'openmicrowave', 'openfridge', 'pickpot')
        
    Returns:
        Path to XML scene file relative to project root
    """
    return f"simulation_assets/scenes/{scene_name}.xml"


def resolve_xml_path(xml_path: str) -> str:
    """
    Resolve XML path to an absolute path.
    If the path doesn't exist as-is, tries to resolve it relative to the project root.
    
    Args:
        xml_path: Path to XML file (can be relative or absolute)
        
    Returns:
        Absolute path to XML file
        
    Raises:
        FileNotFoundError: If the XML file cannot be found
    """
    # If path exists as-is, return it
    if os.path.exists(xml_path):
        return os.path.abspath(xml_path)
    
    # Try to resolve relative to project root
    # Get project root by finding the directory containing 'simulation_assets'
    current_dir = os.path.abspath(os.getcwd())
    project_root = current_dir
    
    # Walk up the directory tree to find project root
    while project_root != os.path.dirname(project_root):
        if os.path.exists(os.path.join(project_root, "simulation_assets")):
            break
        project_root = os.path.dirname(project_root)
    
    # Try path relative to project root
    resolved_path = os.path.join(project_root, xml_path)
    if os.path.exists(resolved_path):
        return os.path.abspath(resolved_path)
    
    # If still not found, raise an error with helpful message
    raise FileNotFoundError(
        f"XML file not found: '{xml_path}'. "
        f"Tried: '{os.path.abspath(xml_path)}' and '{resolved_path}'. "
        f"Current working directory: '{current_dir}'. "
        f"Project root: '{project_root}'"
    )


def get_scene_name_from_xml_path(xml_path: str) -> str:
    """
    Extract scene name from XML file path.
    
    Examples:
        "simulation_assets/scenes/openmicrowave.xml" -> "openmicrowave"
        "simulation_assets/scenes/pickpot.xml" -> "pickpot"
        "ufactory_xarm7/scene.xml" -> "blockstack" (fallback for legacy paths)
    
    Args:
        xml_path: Path to XML file
        
    Returns:
        Scene name string
    """
    # Normalize path
    xml_path = os.path.normpath(xml_path)
    
    # Extract filename without extension
    filename = os.path.basename(xml_path)
    scene_name = os.path.splitext(filename)[0]
    
    # Handle legacy paths that might not match scene names
    if scene_name == "scene":
        # Try to infer from directory structure
        dir_path = os.path.dirname(xml_path)
        if "blockstack" in dir_path.lower() or "mujoco_blockstack" in dir_path.lower():
            return "blockstack"
        # Default to blockstack for legacy compatibility
        return "blockstack"
    
    return scene_name
