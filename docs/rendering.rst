Rendering
=========

.. seealso::

   ``PyQtEnvRendererGL`` — the one OpenGL backend, built on
   ``pyqtgraph.opengl``; :class:`~f1tenth_gym.envs.env_config.RenderConfig` —
   window size, camera target, palette and frame rate.

The renderer is fully decoupled from the physics step: you own the step loop,
and the environment only draws when you call ``env.render()``. Three clocks
therefore run at once — how fast simulated time advances against the wall, how
often a distinct frame is produced, and what frame rate a recorded video
claims — and nearly every question about frames is really a question about
which of the three was meant.

Display setup
-------------

Native Windows uses Qt's ``windows`` platform and PyOpenGL's ``nt`` (WGL)
backend. It does not require ``DISPLAY`` or an X server. Explicit
``QT_QPA_PLATFORM`` and ``PYOPENGL_PLATFORM`` settings take precedence; remove
Linux-only overrides such as ``xcb`` and ``x11`` before running on Windows.

Linux pixel rendering uses ``xcb``/``x11`` and requires ``DISPLAY``. On a
headless Linux server, run under ``xvfb-run -a`` to capture frames. Training
with ``render_enabled=False`` and no render mode requires no display.

The three clocks
----------------

Each has a separate owner, and they never negotiate with one another:

.. list-table::
   :header-rows: 1
   :widths: 22 26 52

   * - Clock
     - Set by
     - Governs
   * - Simulation pace
     - ``real_time_factor``
     - Sim-seconds per wall-second in the human modes; ``1.0`` is real
       time, ``float("inf")`` free-runs. Pacing only — it changes neither
       the physics nor the rgb_array output.
   * - Frame cadence
     - ``render_fps``
     - One distinct rgb_array frame every ``1/render_fps`` *sim*-seconds.
       In the human modes the same number is a wall-clock cap on redraws
       instead.
   * - Video container
     - ``metadata["render_fps"]``
     - The frame rate a recorded file declares. Fixed at
       ``round(1/timestep)``.

All three are readable without a display, because the clock advances on every
``step()`` whether or not a renderer exists:

>>> import gymnasium as gym
>>> import numpy as np
>>> from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig
>>> cfg = EnvConfig(
...     simulation_config=SimulationConfig(max_laps=None), render_enabled=False
... )
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
>>> obs, info = env.reset(seed=42)
>>> env.unwrapped.real_time_factor          # sim seconds per wall second
1.0
>>> env.unwrapped.render_fps                # distinct frames per sim second
60.0
>>> env.metadata["render_fps"]              # recorded-video container fps
100
>>> action = np.array([[0.0, 2.0]], dtype=np.float32)
>>> distinct = 0
>>> for _ in range(100):
...     obs, reward, terminated, truncated, info = env.step(action)
...     distinct += env.unwrapped.frame_is_new
>>> distinct                                # 1.0 sim second of stepping
60
>>> env.unwrapped.set_real_time_factor(5.0) # mid-episode is fine
>>> env.unwrapped.real_time_factor
5.0
>>> env.close()

``set_real_time_factor`` rejects non-positive values, accepts ``inf``, and
re-anchors the pacer so a mid-episode change causes no catch-up sleep or
fast-forward spike. ``frame_is_new`` is the hook for exact-fps capture: append
a frame only on the steps where it is ``True`` and the result is a fixed-rate
video regardless of how the loop is timed.

The env's ``metadata["render_fps"]`` is **not** ``render_config.render_fps``.
It is set to ``round(1/timestep)`` (100 for the default 0.01 s timestep) so
that ``gymnasium.wrappers.RecordVideo`` writes a container that plays back at
real time -- one frame is captured per step. ``render_config.render_fps`` is
the separate display/emit cadence.

Choosing a render mode
----------------------

Two arguments turn drawing on: ``render_enabled=True`` on the
:class:`~f1tenth_gym.envs.env_config.EnvConfig` (the default) and a
``render_mode`` string passed to ``gym.make``. The mode fixes the simulation
pace and decides whether ``env.render()`` returns pixels.

.. list-table::
   :header-rows: 1
   :widths: 22 20 22 36

   * - ``render_mode``
     - Window
     - Simulation pace
     - ``env.render()`` returns
   * - ``"human"``
     - live
     - ``real_time_factor``
     - ``None``
   * - ``"human_fast"``
     - live
     - ``10.0``
     - ``None``
   * - ``"unlimited"``
     - live
     - ``float("inf")``
     - ``None``
   * - ``"rgb_array"``
     - hidden
     - never paces
     - ``(H, W, 3)`` ``uint8``

``"human_fast"`` and ``"unlimited"`` are sugar: they override the configured
``real_time_factor`` at construction, and ``set_real_time_factor`` overrides
them again at any point. Free-running uncaps the *simulation*, not the GPU —
redraws stay capped at ``render_fps`` either way.

.. warning::

   The two switches must agree. ``render_enabled=False`` with
   ``render_mode="human"``, or ``render_enabled=True`` with no ``render_mode``
   at all (the ``gym.make`` default), both leave ``env.unwrapped.renderer`` as
   ``None``: ``render()`` returns ``None`` and ``add_render_callback`` does
   nothing. Check ``env.unwrapped.renderer``, not the config.

A live window is the ordinary loop plus one call:

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig

   cfg = EnvConfig(
       simulation_config=SimulationConfig(max_laps=None),  # default ends at lap 1
   )
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg, render_mode="human")
   obs, info = env.reset(seed=42)

   for _ in range(500):
       obs, reward, terminated, truncated, info = env.step(
           np.array([[0.0, 4.0]], dtype=np.float32)
       )
       env.render()
       if terminated or truncated:
           break
   env.close()

The view is north-up: world +x maps to screen right and +y to up, so it
matches ``plt.plot(xs, ys)`` of the raceline. Cars turn red while colliding
and revert to their palette colour once clear. The mode chosen at ``gym.make``
is the one the backend is built for and it is fixed for the env's lifetime:
the wrappers ``gym.make`` applies reject a ``mode`` keyword outright, and
``env.unwrapped.render(mode="rgb_array")`` on a human-mode env returns
``None`` rather than a frame.

Camera and appearance come from ``RenderConfig`` (:doc:`configuration` lists
every field). Two of them are worth knowing before you reach for them:
``focus_on=None`` parks the camera at the world origin rather than framing the
map — the map view is the renderer's middle-click toggle — and ``show_wheels``
and ``car_thickness`` are stored but never read, so frames are identical
either way.

Why the backend needs a display
-------------------------------

On Linux, the configured Qt/GLX backend needs an X server even for hidden
``rgb_array`` windows; there is no separate offscreen or 2D raster fallback.
Requesting any of the four modes with ``$DISPLAY`` unset on Linux raises a
``RuntimeError`` at construction with setup guidance. Native Windows uses
its desktop Qt/WGL backend and does not perform this X-display check.
A virtual X server satisfies the Linux backend as well as a real one:

.. code-block:: bash

   xvfb-run -a python your_script.py

On Colab, start a virtual display before creating the env, then use
``"rgb_array"`` and embed the frames or the video inline — there is no live
GUI window in a notebook:

.. code-block:: python

   # !apt-get -qq install -y xvfb && pip -q install pyvirtualdisplay
   from pyvirtualdisplay import Display

   Display(visible=0, size=(800, 800)).start()

Linux CI runs the whole test suite under ``xvfb-run``, and the renderer
cases skip themselves on Linux when ``$DISPLAY`` is unset (:doc:`installation`).
On native Windows the renderer cases use the desktop display.

Grabbing frames and recording video
-----------------------------------

In ``"rgb_array"`` mode ``env.render()`` never sleeps and always hands back a
contiguous ``(H, W, 3)`` ``uint8`` array pinned to the square ``window_size``,
whatever the display's pixel ratio. Between distinct frames it returns the
cached one, so ``frame_is_new`` is what separates real frames from repeats:

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig, RenderConfig, SimulationConfig

   cfg = EnvConfig(
       simulation_config=SimulationConfig(max_laps=None),
       render_config=RenderConfig(window_size=600),
   )
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg, render_mode="rgb_array")
   obs, info = env.reset(seed=42)

   frames = []
   for _ in range(100):
       obs, reward, terminated, truncated, info = env.step(
           np.array([[0.0, 4.0]], dtype=np.float32)
       )
       if env.unwrapped.frame_is_new:
           frames.append(env.render())
       if terminated or truncated:
           break
   env.close()

   print(len(frames), frames[0].shape, frames[0].dtype)

That prints ``60 (600, 600, 3) uint8`` — one second of simulated time at the
default 60 fps cadence, with the cars drawn into every frame.

``gymnasium.wrappers.RecordVideo`` needs no cadence logic of its own: it
captures once per step and the container fps is the env's
``metadata["render_fps"]``, which is why a 2.0 sim-second episode becomes a
2.0-second file:

.. code-block:: python

   from gymnasium.wrappers import RecordVideo

   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg, render_mode="rgb_array")
   env = RecordVideo(env, video_folder="videos")
   # ... reset / step loop, then:
   env.close()   # writes videos/rl-video-episode-0.mp4

Encoding goes through moviepy, which arrives with the ``examples`` dependency
group that ``uv sync`` installs by default. Sync with ``--no-group examples``
and ``RecordVideo`` raises
``gymnasium.error.DependencyNotInstalled`` instead; :doc:`installation` covers
what else that sync prunes.

Drawing your own geometry
-------------------------

``env.unwrapped.add_render_callback(fn)`` registers a
``fn(env_renderer) -> None`` invoked once per ``render()``, before the cars
are drawn. The callback reads the last observation from ``env_renderer.obs``
and builds persistent geometry through the renderer's three factories —
``get_points_renderer``, ``get_lines_renderer`` and
``get_closed_lines_renderer`` — each of which returns an object with an
``update(points)`` method to call on later frames.

``make_lidar_scan_callback`` is a ready-made one that converts an agent's
scan into a world-frame point cloud:

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig
   from f1tenth_gym.envs.rendering import make_lidar_scan_callback

   cfg = EnvConfig(simulation_config=SimulationConfig(max_laps=None))
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg, render_mode="human")
   env.unwrapped.add_render_callback(
       make_lidar_scan_callback(
           "agent_0", cfg.lidar_config, color=(255, 0, 0), size=3, subsample=4
       )
   )

   obs, info = env.reset(seed=42)
   for _ in range(500):
       obs, reward, terminated, truncated, info = env.step(
           np.array([[0.0, 4.0]], dtype=np.float32)
       )
       env.render()
       if terminated or truncated:
           break
   env.close()

It takes the environment's :class:`~f1tenth_gym.envs.lidar.LiDARConfig`
because it precomputes beam angles from ``angle_min``, ``angle_max`` and
``num_beams`` and applies the sensor mounting offset; ``subsample`` draws
every Nth beam. With ``lidar_config.enabled=False`` there is no ``scan`` field
to read and the callback draws nothing.

.. note::

   Callbacks always see the ``DEFAULT`` field vocabulary, whatever
   ``ObservationConfig.type`` the env was built with: ``F110Env`` keeps a
   second, permanently-``DEFAULT`` observation type just for the renderer. A
   callback under ``KINEMATIC_STATE`` therefore finds ``std_state`` and
   ``scan`` in ``env_renderer.obs`` but raises ``KeyError`` on ``pose_x``
   (:doc:`observations`).

Callbacks live on the renderer, not on the env, so
``env.unwrapped.configure(...)`` builds a fresh renderer with an empty
callback list — re-register after any reconfiguration
(:doc:`configuration`). ``examples/waypoint_follow.py`` drives the human
renderer with a planner's own callbacks end to end (:doc:`examples`).
