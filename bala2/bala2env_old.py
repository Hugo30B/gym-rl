"""
Bala2Env — Digital Twin of the M5Stack Bala2Fire self-balancing robot.

Based on the CartPole-v1 environment by Rich Sutton et al., rewritten from
scratch to close the Sim-to-Real gap for a two-wheeled ESP32 robot:

  1. Continuous action space (PWM motors)  →  spaces.Box(-1, 1)
  2. Linear friction on the wheel axis     →  friction_coef = 0.5
  3. Real physical parameters of Bala2Fire →  masses, length, tau @ 50 Hz
  4. Reward shaping                        →  penalises motor effort & speed
  5. Restyled renderer                     →  dark-grey wheels + orange chassis
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class Bala2Env(gym.Env):
    """
    ## Description
    Digital twin of the M5Stack Bala2Fire, a two-wheeled self-balancing robot
    based on ESP32 running a 50 Hz control loop.

    The underlying physics follow the classic cart-pole model (Barto, Sutton &
    Anderson, 1983) with real-world corrections: continuous PWM actions,
    wheel-axis friction, and hardware-accurate inertial parameters.

    ## Action Space
    Box(low=-1.0, high=1.0, shape=(1,), dtype=float32)
    The scalar represents normalised motor effort.  Internally it is scaled by
    `force_mag` (5.0 N) to obtain the physical force applied to the wheel axis.

    ## Observation Space
    Box with shape (4,):
        [0] cart position x          (m)     in [-4.8,  4.8]
        [1] cart velocity x_dot      (m/s)   in (-inf, +inf)
        [2] pole angle theta         (rad)   in [-0.418, 0.418]
        [3] pole angular velocity    (rad/s) in (-inf, +inf)

    ## Reward
        +1.0  every surviving step
        -0.1 * |action|    motor-effort penalty   (discourages vibrations)
        -0.05 * |x_dot|    speed penalty           (encourages stillness)
        -1.0  on termination (robot fell or left track)

    ## Episode End
        Termination : |theta| > 12 deg  or  |x| > 2.4 m
        Truncation  : step count > 500  (handled by TimeLimit wrapper)

    ## Physical Parameters (M5Stack Bala2Fire)
        masscart       = 0.15 kg   (wheels + motors)
        masspole       = 0.25 kg   (body with PCB + battery)
        length         = 0.06 m    (distance to centre of mass)
        force_mag      = 5.0  N    (motor force scale)
        tau            = 0.02 s    (50 Hz — matches ESP32 control loop)
        friction_coef  = 0.5       (wheel-axis viscous friction coefficient)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #

    def __init__(self, render_mode: str | None = None):

        # --- MOD 3: Physical constants of the M5Stack Bala2Fire -------- #
        self.gravity          = 9.8
        self.masscart         = 0.15   # kg  — wheels + motors
        self.masspole         = 0.25   # kg  — M5Stack body + battery
        self.total_mass       = self.masscart + self.masspole   # 0.40 kg
        self.length           = 0.06   # m   — half-length / CoM distance
        self.polemass_length  = self.masspole * self.length     # 0.015 kg·m
        self.force_mag        = 5.0    # N   — motor force scale
        self.tau              = 0.02   # s   — 50 Hz control loop (ESP32)

        # --- MOD 2: Viscous wheel-axis friction ------------------------ #
        self.friction_coef    = 0.5    # N·s/m

        # --- Termination thresholds ------------------------------------ #
        self.theta_threshold_radians = 12 * 2 * math.pi / 360   # ±12°
        self.x_threshold             = 2.4                       # ±2.4 m

        # --- MOD 1: Continuous action space (normalised PWM) ----------- #
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        obs_high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # --- Renderer -------------------------------------------------- #
        self.render_mode  = render_mode
        self.screen_width  = 600
        self.screen_height = 400
        self.screen        = None
        self.clock         = None
        self.isopen        = True

        # --- Internal state -------------------------------------------- #
        self.state: np.ndarray | None = None
        self.steps_beyond_terminated: int | None = None

    # ------------------------------------------------------------------ #
    #  Step                                                               #
    # ------------------------------------------------------------------ #

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action), (
            f"{action!r} ({type(action)}) invalid"
        )
        assert self.state is not None, "Call reset() before step()."

        x, x_dot, theta, theta_dot = self.state

        # MOD 1 — Scale and clip normalised action to physical force
        raw   = float(np.clip(action, -1.0, 1.0)[0])
        force = raw * self.force_mag

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # Standard inverted-pendulum equations of motion
        # Reference: https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass

        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # MOD 2 — Subtract viscous friction proportional to current velocity
        xacc = xacc - self.friction_coef * x_dot

        # Standard Euler integration
        x         = x         + self.tau * x_dot
        x_dot     = x_dot     + self.tau * xacc
        theta     = theta     + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)

        terminated = bool(
            x     < -self.x_threshold
            or x     >  self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta >  self.theta_threshold_radians
        )

        # MOD 4 — Reward shaping: base reward minus effort & speed penalties
        if not terminated:
            if self.steps_beyond_terminated is not None:
                if self.steps_beyond_terminated == 0:
                    logger.warn(
                        "step() called after terminated=True. "
                        "Always call reset() when you receive terminated=True."
                    )
                self.steps_beyond_terminated += 1

            reward = (
                1.0
                - 0.1  * abs(raw)     # motor-effort penalty
                - 0.05 * abs(x_dot)   # speed penalty
            )
        else:
            if self.steps_beyond_terminated is None:
                self.steps_beyond_terminated = 0
            reward = -1.0

        if self.render_mode == "human":
            self.render()

        # truncation=False: handled by the TimeLimit wrapper in gym.make()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    # ------------------------------------------------------------------ #
    #  Reset                                                              #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        low, high = utils.maybe_parse_reset_bounds(options, -0.05, 0.05)
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), {}

    # ------------------------------------------------------------------ #
    #  Render  (MOD 5 — Bala2Fire visual style)                          #
    # ------------------------------------------------------------------ #

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "render() called without a render_mode. "
                f'Use gym.make("{self.spec.id}", render_mode="rgb_array").'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is required: pip install "gymnasium[classic-control]"'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("Bala2Fire — Digital Twin")
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state is None:
            return None

        x_state = self.state

        # World-to-screen scale
        world_width = self.x_threshold * 2
        scale       = self.screen_width / world_width

        # ---- MOD 5: Bala2Fire geometry & colours ---------------------- #
        #
        #   Wheels / base  — dark grey (#323232), narrower & flatter than CartPole
        #   Chassis body   — M5Stack orange (#FF7800), wider & shorter than CartPole pole
        #   Axle pin       — medium grey (#646464)
        #
        cart_w   = 36.0              # px  width  of wheel assembly
        cart_h   = 18.0              # px  height of wheel assembly
        pole_w   = 20.0              # px  width  of M5Stack chassis
        pole_len = scale * (2 * self.length) * 4.5  # px length, scaled for visibility

        COLOR_BG      = (240, 240, 240)
        COLOR_WHEELS  = ( 50,  50,  50)   # dark grey — wheel pair
        COLOR_CHASSIS = (255, 120,   0)   # M5Stack orange — body
        COLOR_AXLE    = (100, 100, 100)   # grey axle pin
        COLOR_TRACK   = (  0,   0,   0)   # ground line

        surf = pygame.Surface((self.screen_width, self.screen_height))
        surf.fill(COLOR_BG)

        cart_x     = x_state[0] * scale + self.screen_width / 2.0
        cart_y     = 120  # y-coordinate of the top of the wheel block
        axle_offset = cart_h / 4.0

        # -- Wheel assembly (cart) --
        l = -cart_w / 2;  r = cart_w / 2
        t =  cart_h / 2;  b = -cart_h / 2
        cart_coords = [
            (l + cart_x, b + cart_y),
            (l + cart_x, t + cart_y),
            (r + cart_x, t + cart_y),
            (r + cart_x, b + cart_y),
        ]
        gfxdraw.aapolygon(surf, cart_coords, COLOR_WHEELS)
        gfxdraw.filled_polygon(surf, cart_coords, COLOR_WHEELS)

        # -- M5Stack chassis body (pole) — orange rectangle, rotated --
        lp = -pole_w / 2
        rp =  pole_w / 2
        tp =  pole_len - pole_w / 2
        bp = -pole_w / 2

        pole_coords = []
        for coord in [(lp, bp), (lp, tp), (rp, tp), (rp, bp)]:
            vec   = pygame.math.Vector2(coord).rotate_rad(-x_state[2])
            point = (vec[0] + cart_x, vec[1] + cart_y + axle_offset)
            pole_coords.append(point)

        gfxdraw.aapolygon(surf, pole_coords, COLOR_CHASSIS)
        gfxdraw.filled_polygon(surf, pole_coords, COLOR_CHASSIS)

        # -- Axle pin --
        axle_r = int(pole_w / 2)
        gfxdraw.aacircle(
            surf, int(cart_x), int(cart_y + axle_offset), axle_r, COLOR_AXLE
        )
        gfxdraw.filled_circle(
            surf, int(cart_x), int(cart_y + axle_offset), axle_r, COLOR_AXLE
        )

        # -- Ground track line --
        gfxdraw.hline(surf, 0, self.screen_width, cart_y, COLOR_TRACK)

        # Flip vertically (pygame Y-axis is inverted relative to physics)
        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    # ------------------------------------------------------------------ #
    #  Close                                                              #
    # ------------------------------------------------------------------ #

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False