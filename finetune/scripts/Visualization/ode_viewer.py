"""ODE drawstuff OpenGL 3D viewer — volumetric body segments via drawstuff.

Usage:
  from Visualization.ode_viewer import ODEViewer
  viewer = ODEViewer(env)          # configure camera + hide T-pose ref
  viewer.start()                   # launch GLUT window
  for each frame: viewer.tick()    # pace playback
  viewer.hold()                    # keep window open after sim
"""

import os

_MOCONVQ_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MoConVQ')


class ODEViewer:

    def __init__(self, env):
        from VclSimuBackend import (
            visSetWorld, visLookAt, visTrackBody,
            visSetJointRadius, visSetAxisLength,
            visWhetherHingeAxis, visWhetherLocalAxis,
        )

        self._env = env
        self._scene = env.scene
        self._character = env.sim_character
        self._started = False
        self._frame = 0
        self._pause_ms = 50  # ~20fps

        # --- Hide the kinematic reference T-pose character ---
        # Move its ENTIRE body chain far underground.
        # The root body Position setter only moves root; children stay relative.
        # We move every body individually.
        kin = env.scene.characters[1]
        for body in kin.body_info.bodies:
            pos = body.PositionNumpy.copy()
            pos[1] = -1000.0  # far underground
            body.Position = pos

        # --- Character: light blue ---
        for body in self._character.body_info.bodies:
            for geom in body.geom_iter():
                geom.render_by_default_color = 0
                import numpy as np
                geom.render_user_color = np.array([0.4, 0.7, 1.0])  # light blue

        # --- ODE world binding ---
        visSetWorld(self._scene.world)

        # --- Clean rendering ---
        visSetJointRadius(0.02)
        visSetAxisLength(0.0)
        visWhetherHingeAxis(0)
        visWhetherLocalAxis(0)
        # Keep background ON (floor + sky), just hide joint axes

        # --- Camera: fixed, character-centered ---
        # Character spawns near world origin. Fixed camera avoids tracking
        # issues; mouse rotation/zoom naturally orbits the target point.
        visLookAt(
            pos=[2.2, 2.2, 3.5],       # front-right, slightly above
            target=[0.0, 1.0, 0.0],     # hip height at origin = character center
            up=[0.0, 1.0, 0.0],
        )

    def start(self):
        if self._started:
            return
        from VclSimuBackend import visDrawWorld
        os.chdir(_MOCONVQ_ROOT)
        print('[ODEViewer] Starting OpenGL render window...')
        visDrawWorld()
        self._started = True
        self._frame = 0

    def tick(self):
        if not self._started:
            return
        from VclSimuBackend import visPause
        self._frame += 1
        visPause(self._pause_ms)
        if self._frame % 60 == 0:
            print(f'[ODEViewer]  Frame {self._frame}')

    def hold(self):
        """Keep window open. Press Ctrl+C or close GLUT window to exit."""
        if not self._started:
            return
        print('[ODEViewer] Done. Window stays open — Ctrl+C to close.')
        try:
            while True:
                from VclSimuBackend import visPause
                visPause(200)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        if not self._started:
            return
        from VclSimuBackend import visKill
        print('[ODEViewer] Closing.')
        visKill()
        self._started = False
