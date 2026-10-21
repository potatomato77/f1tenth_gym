import os
import sys
from typing import Optional, TYPE_CHECKING

from .renderer import EnvRenderer, ObjectRenderer
from .callbacks import make_lidar_scan_callback
from ..dynamic_models import VehicleParameters

if TYPE_CHECKING:
    from ..track import Track
    from ..env_config import RenderConfig

__all__ = [
    "EnvRenderer",
    "ObjectRenderer",
    "make_lidar_scan_callback",
    "make_renderer",
]

NO_DISPLAY_GUIDANCE = (
    "F1TENTH rendering needs an X display (real or virtual), but no $DISPLAY was found.\n"
    "  • Headless server: run your script under a virtual display:\n"
    "        xvfb-run -a python your_script.py\n"
    "  • Google Colab (in a cell, before creating the env):\n"
    "        !apt-get -qq install -y xvfb\n"
    "        !pip -q install pyvirtualdisplay\n"
    "        from pyvirtualdisplay import Display\n"
    "        Display(visible=0, size=(800, 800)).start()\n"
    "    then use render_mode='rgb_array' and embed the frames/video inline."
)

# Render modes that draw pixels and therefore require a desktop display. (gymnasium
# strips the "_list" suffix, so the env only ever sees "rgb_array" here.)
_DISPLAY_RENDER_MODES = ("human", "human_fast", "unlimited", "rgb_array")


def make_renderer(
    params: VehicleParameters,
    track: "Track",
    agent_ids: list[str],
    render_mode: Optional[str] = None,
    render_config: "RenderConfig" = None,
) -> tuple[EnvRenderer, "RenderConfig"]:
    """Return a GL renderer and the RenderConfig it was built from.

    Windows uses its native Qt/WGL backend without an X display. Other platforms
    require ``$DISPLAY`` and raise setup guidance before constructing Qt objects.
    """
    needs_display = render_mode in _DISPLAY_RENDER_MODES
    is_windows = sys.platform == "win32"
    if needs_display and not is_windows and not os.environ.get("DISPLAY"):
        raise RuntimeError(NO_DISPLAY_GUIDANCE)

    if render_config is None:
        from ..env_config import RenderConfig  # local import avoids an import cycle
        render_config = RenderConfig()

    if render_mode in ["human", "rgb_array", "unlimited", "human_fast"]:
        # Qt and PyOpenGL must agree on the windowing API, else the cars vanish
        # from rgb_array frames. Both pins must precede the first GL import below.
        if is_windows:
            os.environ.setdefault("QT_QPA_PLATFORM", "windows")
            os.environ.setdefault("PYOPENGL_PLATFORM", "nt")
        else:
            os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
            os.environ.setdefault("PYOPENGL_PLATFORM", "x11")
        # NVIDIA ignores the backend's swap interval 0 and holds swapBuffers to the
        # vblank, pinning human modes to the monitor refresh: 59.9 vs 1051 Hz.
        os.environ.setdefault("__GL_SYNC_TO_VBLANK", "0")

        from .rendering_pyqtgl import PyQtEnvRendererGL as EnvRenderer

        renderer = EnvRenderer(
            params=params,
            track=track,
            agent_ids=agent_ids,
            render_config=render_config,
            render_mode=render_mode,
        )
    else:
        renderer = None
    return renderer, render_config
