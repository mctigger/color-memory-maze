# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

### Conda Environment
```bash
# Activate the correct conda environment
conda activate memorymaze-drstrategy
```

## Commands

### Testing and Development
```bash
# No formal test suite - use example scripts for validation
python examples/save_50_observations.py --env-id MemoryMaze-9x9-v0
python examples/compare_observations.py
python gui/run_gui.py  # Interactive GUI for testing

# Data generation scripts (in data/ directory)
python data/single_room_reset_3x3_new.py
python data/four_rooms_reset_7x7_two_traversals_new.py
```

### Development Setup
```bash
# Install in development mode
pip install -e .
pip install -e .[dev]  # Includes development tools

# Code formatting and linting (configured in pyproject.toml)
black src/ examples/ gui/ data/ --line-length 88
isort src/ examples/ gui/ data/ --profile black
flake8 src/ examples/ gui/ data/
```

### Package Building
```bash
# Build package using uv
python -m build
twine upload dist/*  # Publishing
```

### Environment Testing
```bash
# Set rendering backend (required for headless environments)
export MUJOCO_GL=egl     # Hardware accelerated (recommended)
export MUJOCO_GL=osmesa  # Software rendering for headless servers

# Test environment creation
python -c "import gymnasium as gym; import memory_maze; env = gym.make('MemoryMaze-9x9-v0'); obs, info = env.reset(); print(f'Success: {obs.shape}'); env.close()"
```

## Architecture and Core Components

### Package Structure
- **`src/memory_maze/`**: Core package with modern gymnasium integration
  - `__init__.py`: Environment registration system for all maze variants
  - `tasks.py`: Task definitions and environment factory functions
  - `maze.py`: Core maze generation and layout logic
  - `gym_wrappers.py`: Gymnasium compatibility layer
  - `custom_task.py`: Extended task definitions with custom layouts
  - `drstrategy_2d_envs.py`: DrStrategy-specific 2D environment configurations

### Environment Categories
1. **Original Memory Maze**: Standard 9x9 to 15x15 scavenger hunt environments
2. **DrStrategy Custom**: Enhanced environments with complex mazes and rich visual textures
3. **Multi-room Layouts**: Single room (3x3) to twenty rooms (7x39) configurations

### Key Environment IDs
```python
# Original Memory Maze (with variants: -HD, -Vis, -Top, -ExtraObs, -Oracle, -HiFreq)
'MemoryMaze-9x9-v0', 'MemoryMaze-11x11-v0', 'MemoryMaze-13x13-v0', 'MemoryMaze-15x15-v0'

# DrStrategy Custom Environments
'MemoryMaze-single-room-3x3-v0'
'MemoryMaze-two-rooms-3x7-v0', 'MemoryMaze-two-rooms-3x7-fixed-layout-v0'
'MemoryMaze-four-rooms-7x7-fixed-layout-v0'
'MemoryMaze-cmaze-7x7-fixed-layout-v0'  # Rich colorful textures
'MemoryMaze-cmaze-7x7-consistent-target-v0'  # Consistent visual target generation
'MemoryMaze-cmaze-15x15-fixed-layout-v0'  # Complex maze with enhanced visuals
'MemoryMaze-twenty-rooms-7x39-fixed-layout-random-goals-v0'
```

### Rendering and Observation System
- **Resolution**: Default 64x64, HD variants at 256x256, flexible resolution support
- **Format**: RGB images in HWC format (Height x Width x Channels)
- **Backends**: EGL (hardware, recommended), OSMesa (software), GLFW (windowed)
- **Camera Views**: First-person (default) and top-down variants available

### DrStrategy Enhancements
The key differentiator of this package is the integration of DrStrategy improvements:
- **Visual Complexity**: Rich, colorful wall textures vs. standard yellow walls
- **Layout Variety**: Complex mazes with multiple pathways and dead ends
- **Fixed Layouts**: Reproducible maze structures for controlled experiments
- **Texture Discrimination**: Environments designed to test visual pattern recognition
- **Consistent Target Generation**: Visual targets based on actual goal positions (not random)

### Wrapper Architecture (Restructured)
The environment construction follows this optimized wrapper chain:
```
composer.Environment (MuJoCo physics)
  ↓
GymWrapper (Applied immediately after composer.Environment)
  ↓ 
Gymnasium wrappers (in same order):
  ├── GymRemapObservationWrapper
  ├── GymTargetColorAsBorderWrapper  
  ├── GymImageOnlyObservationWrapper
  └── GymDiscreteActionSetWrapper
  ↓
GymConsistentTargetWrapper (for consistent target environments)
```
This structure applies the Gymnasium interface as early as possible while maintaining compatibility.

### Data Generation Pipeline
Located in `data/` directory with environment-specific generation scripts:
- Naming pattern: `{layout}_reset_{size}_{variant}_new.py`
- Includes shell scripts for batch processing
- Supports trajectory generation and observation capture

### Example Scripts
- **`examples/save_50_observations.py`**: Environment validation and observation capture
- **`examples/compare_observations.py`**: Visual comparison between environment variants
- **`examples/keyboard_navigation.py`**: Interactive navigation for development
- **`gui/run_gui.py`**: Full GUI application for environment interaction

### Dependencies and Compatibility
- **Core**: gymnasium (NOT legacy gym), dm_control, mujoco>=3.3.5
- **Rendering**: Requires proper MUJOCO_GL configuration
- **Python**: 3.8+ support, optimized for 3.10+
- **Packaging**: Modern pyproject.toml with uv build backend

### Environment Registration System
All environments auto-register on import via `memory_maze.__init__.py`. The system supports:
- Flexible resolution through custom wrapper classes
- Variant generation (HD, Top-down, Oracle, etc.)
- Multiple observation space configurations (image-only vs. global observables)