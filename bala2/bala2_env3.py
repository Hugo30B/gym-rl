"""
Bala2Env — Digital Twin of the M5Stack Bala2Fire self-balancing robot.

Modificaciones para Sim-to-Real:
  - Fricción reducida para mayor realismo.
  - Ruido en observaciones (IMU Noise).
  - Penalización por posición (anti-deriva).
  - Truncación manual a 500 pasos.
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
    Digital twin of the M5Stack Bala2Fire. Refined for Sim-to-Real gap.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: str | None = None):

        # --- Physical constants ---
        self.gravity          = 9.8
        self.masscart         = 0.15   
        self.masspole         = 0.25   
        self.total_mass       = self.masscart + self.masspole   
        self.length           = 0.06   
        self.polemass_length  = self.masspole * self.length     
        self.force_mag        = 5.0    
        self.tau              = 0.02   

        # --- MODIFICACIÓN: Fricción más baja para que el robot sea inestable ---
        self.friction_coef    = 0.05    

        # --- MODIFICACIÓN: Ruido de sensor (Sim-to-Real) ---
        self.obs_noise        = 0.01    # Desviación estándar del ruido

        # --- Límites ---
        self.theta_threshold_radians = 12 * 2 * math.pi / 360   
        self.x_threshold             = 2.4                       
        self.max_episode_steps       = 500

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        obs_high = np.array(
            [self.x_threshold * 2, np.inf, self.theta_threshold_radians * 2, np.inf],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.render_mode  = render_mode
        self.screen_width  = 600
        self.screen_height = 400
        self.screen        = None
        self.clock         = None
        self.isopen        = True

        self.state: np.ndarray | None = None
        self.steps_beyond_terminated: int | None = None
        self.elapsed_steps = 0

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action), f"{action!r} invalid"
        assert self.state is not None, "Call reset() before step()."

        self.elapsed_steps += 1
        x, x_dot, theta, theta_dot = self.state

        raw   = float(np.clip(action, -1.0, 1.0)[0])
        force = raw * self.force_mag

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        xacc = xacc - self.friction_coef * x_dot

        # Euler integration
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

        # Truncación manual para evitar recompensas infinitas
        truncated = self.elapsed_steps >= self.max_episode_steps

        if not terminated:
            # --- MODIFICACIÓN: Reward shaping mejorado ---
            reward = (
                1.0
                - 0.1  * abs(raw)     # penalizar esfuerzo motor
                - 0.05 * abs(x_dot)   # penalizar velocidad
                - 0.1  * abs(x)       # penalizar alejarse del centro (EVITA DERIVA)
            )
        else:
            reward = -1.0

        # --- MODIFICACIÓN: Añadir ruido a la observación antes de devolverla ---
        noise = self.np_random.normal(0, self.obs_noise, size=(4,)).astype(np.float32)
        obs_with_noise = np.array(self.state, dtype=np.float32) + noise

        if self.render_mode == "human":
            self.render()

        return obs_with_noise, reward, terminated, truncated, {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.elapsed_steps = 0
        low, high = utils.maybe_parse_reset_bounds(options, -0.05, 0.05)
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        # El código del renderizado se mantiene intacto para no romper la visualización
        if self.render_mode is None:
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled('pygame is required') from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("Bala2Fire — Digital Twin")
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state is None:
            return None

        x_state = self.state
        theta   = float(x_state[2])
        world_width = self.x_threshold * 2
        scale       = self.screen_width / world_width
        r_wheel  = 28
        y_ground = 80           
        pivot_x  = int(x_state[0] * scale + self.screen_width / 2.0)
        pivot_y  = y_ground + r_wheel   

        COLOR_BG         = (245, 245, 248)
        COLOR_GROUND_FILL= (210, 210, 210)
        COLOR_GROUND_LINE= ( 60,  60,  60)
        COLOR_TYRE       = ( 35,  35,  35)   
        COLOR_RIM        = ( 90,  90,  90)   
        COLOR_SPOKE      = (130, 130, 130)   
        COLOR_HUB        = ( 50,  50,  50)   
        COLOR_SHADOW_WHEEL=(110, 110, 110)   
        COLOR_STEM       = ( 75,  75,  75)   
        COLOR_BODY       = (255, 120,   0)   
        COLOR_BODY_SHADE = (200,  88,   0)   
        COLOR_LCD        = ( 20, 130, 230)   
        COLOR_LCD_DARK   = ( 10,  80, 160)   
        COLOR_BTN        = (215, 215, 215)   

        surf = pygame.Surface((self.screen_width, self.screen_height))
        surf.fill(COLOR_BG)

        def rot(local_x, local_y_up):
            vec = pygame.math.Vector2(local_x, local_y_up).rotate_rad(-theta)
            return (vec.x + pivot_x, vec.y + pivot_y)

        def quad(corners):
            return [rot(lx, ly) for lx, ly in corners]

        stem_w, stem_h = 10, 50
        box_w, box_h = 66, 32
        lcd_w, lcd_h, lcd_y_bot = 38, 20, stem_h + 5
        btn_w, btn_h, btn_y_bot = 10, 6, stem_h + box_h - 3
        btn_xs = [-20, 0, 20]

        pygame.draw.rect(surf, COLOR_GROUND_FILL, (0, 0, self.screen_width, y_ground))
        gfxdraw.hline(surf, 0, self.screen_width, y_ground, COLOR_GROUND_LINE)

        bx, by = pivot_x - 6, pivot_y + 3
        gfxdraw.filled_circle(surf, bx, by, r_wheel, COLOR_SHADOW_WHEEL)
        gfxdraw.aacircle(surf, bx, by, r_wheel, COLOR_SHADOW_WHEEL)

        stem_pts = quad([(-stem_w/2, 0), (-stem_w/2, stem_h), (stem_w/2, stem_h), (stem_w/2, 0)])
        gfxdraw.filled_polygon(surf, stem_pts, COLOR_STEM)
        gfxdraw.aapolygon(surf, stem_pts, COLOR_STEM)

        body_pts = quad([(-box_w/2, stem_h), (-box_w/2, stem_h+box_h), (box_w/2, stem_h+box_h), (box_w/2, stem_h)])
        gfxdraw.filled_polygon(surf, body_pts, COLOR_BODY)
        gfxdraw.aapolygon(surf, body_pts, COLOR_BODY)

        lcd_pts = quad([(-lcd_w/2, lcd_y_bot), (-lcd_w/2, lcd_y_bot+lcd_h), (lcd_w/2, lcd_y_bot+lcd_h), (lcd_w/2, lcd_y_bot)])
        gfxdraw.filled_polygon(surf, lcd_pts, COLOR_LCD_DARK)
        gfxdraw.aapolygon(surf, lcd_pts, COLOR_LCD_DARK)

        for bx_loc in btn_xs:
            btn_pts = quad([(bx_loc-btn_w/2, btn_y_bot-btn_h), (bx_loc-btn_w/2, btn_y_bot), (bx_loc+btn_w/2, btn_y_bot), (bx_loc+btn_w/2, btn_y_bot-btn_h)])
            gfxdraw.filled_polygon(surf, btn_pts, COLOR_BTN)

        gfxdraw.filled_circle(surf, pivot_x, pivot_y, r_wheel, COLOR_TYRE)
        gfxdraw.aacircle(surf, pivot_x, pivot_y, r_wheel, COLOR_TYRE)
        gfxdraw.filled_circle(surf, pivot_x, pivot_y, 6, COLOR_HUB)

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False