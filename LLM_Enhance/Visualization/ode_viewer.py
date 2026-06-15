"""ODE drawstuff OpenGL 3D viewer — volumetric body segments via drawstuff.

Usage:
  from LLM_Enhance.Visualization.ode_viewer import ODEViewer
  viewer = ODEViewer(env, center=root_pos)   # center on character
  viewer.start()                              # launch GLUT window
  viewer.start_recording()                    # start screen capture
  for each frame: viewer.tick()
  viewer.save_gif('output.gif')               # save recorded frames + close
"""

import os
import numpy as np

_MOCONVQ_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'MoConVQ')


class ODEViewer:

    def __init__(self, env, center=None):
        """center: (x, y, z) world position to point camera at. Default: origin."""
        from VclSimuBackend import (
            visSetWorld, visLookAt, visSetJointRadius, visSetAxisLength,
            visWhetherHingeAxis, visWhetherLocalAxis,
        )

        self._env = env
        self._scene = env.scene
        self._character = env.sim_character
        self._started = False
        self._recording = False
        self._frame = 0
        self._pause_ms = 50

        if center is None:
            center = (0.0, 1.0, 0.0)
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

        # --- Hide kinematic T-pose character underground ---
        kin = env.scene.characters[1]
        for body in kin.body_info.bodies:
            pos = body.PositionNumpy.copy()
            pos[1] = -1000.0
            body.Position = pos

        # --- Light blue character ---
        for body in self._character.body_info.bodies:
            for geom in body.geom_iter():
                geom.render_by_default_color = 0
                geom.render_user_color = np.array([0.4, 0.7, 1.0])

        # --- ODE world binding ---
        visSetWorld(self._scene.world)

        # --- Clean appearance ---
        visSetJointRadius(0.02)
        visSetAxisLength(0.0)
        visWhetherHingeAxis(0)
        visWhetherLocalAxis(0)

        # --- Camera centered on character's actual spawn position ---
        visLookAt(
            pos=[cx + 1.5, cy + 5.0, cz + 4.0],   # above-front of character
            target=[cx, cy + 0.9, cz],              # hip height
            up=[0.0, 1.0, 0.0],
        )

    # ── lifecycle ──────────────────────────────────────────────────

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

    # ── screen recording ───────────────────────────────────────────

    def start_recording(self):
        """Begin capturing drawstuff frames into an internal buffer."""
        from VclSimuBackend import visStartRecordVideo
        visStartRecordVideo()
        self._recording = True
        print('[ODEViewer] Recording started.')

    def stop_recording(self):
        """Stop recording and return captured frames (N, H, W, 3) BGR."""
        from VclSimuBackend import visEndRecordVideo
        if self._recording:
            frames = visEndRecordVideo()
            self._recording = False
            print(f'[ODEViewer] Captured {len(frames)} frames.')
            return frames
        return None

    def save_gif(self, output_path, fps=20, max_frames=200):
        """Stop recording, subsample, save as GIF, close viewer."""
        frames = self.stop_recording()
        self.close()

        if frames is not None and len(frames) > 0:
            # Subsample to avoid huge files (recorder runs at ~60fps)
            step = max(1, len(frames) // max_frames)
            frames = frames[::step]
            print(f'[ODEViewer] Subsampled to {len(frames)} frames (step={step}).')
            self._write_gif(frames, output_path, fps)

    def _write_gif(self, frames, output_path, fps):
        """Convert numpy frame array (N, H, W, 3) BGR to GIF via PIL."""
        from PIL import Image
        images = []
        for i in range(len(frames)):
            # BGR → RGB, flip vertically (OpenGL reads bottom-up)
            rgb = frames[i][::-1, :, ::-1]
            img = Image.fromarray(rgb)
            images.append(img.resize((img.width // 2, img.height // 2), Image.LANCZOS))

        images[0].save(
            output_path, save_all=True, append_images=images[1:],
            duration=int(1000 / fps), loop=0,
        )
        print(f'[ODEViewer] Saved: {output_path}  ({len(images)} frames, {fps} fps)')
