import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

class Bala2Env(gym.Env):
    """
    Digital twin of the M5Stack Bala2Fire — Simplified & Effective Sim-to-Real.
    Corregido con dinámica de cuerpo rígido (Caja), momentos de inercia reales,
    y penalización de suavidad (anti-bang-bang) para evitar temblores físicos.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: str | None = None):
        # ── Constantes físicas nominales ───────────────────────────────────
        self.gravity = 9.8
        self.masscart_nom = 0.161    # kg (Ruedas + chasis bajo)
        self.masspole_nom = 0.124    # kg (Cuerpo superior)
        
        # Dimensiones reales medidas
        self.box_height_nom = 0.045  # m (Altura desde el eje hasta arriba)
        self.box_depth_nom  = 0.050  # m (2.5cm hacia adelante + 2.5cm hacia atrás)
        
        self.force_mag = 5.0     # N máximo aplicable
        self.tau = 0.005         # s (200 Hz)

        # ── Límites ────────────────────────────────────────────────────────
        self.theta_threshold_radians = 40 * 2 * math.pi / 360  # ±40°
        self.x_threshold = 2.4  # m
        self.max_episode_steps = 2000

        # ── Motor y Ruido Nominal ──────────────────────────────────────────
        self.motor_deadzone = 0.05
        self.motor_lag_alpha = 0.8  # 0.0 = sin lag, 1.0 = congelado

        # ── Espacios ───────────────────────────────────────────────────────
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        obs_high = np.array(
            [self.x_threshold * 2, np.inf, self.theta_threshold_radians * 2, np.inf],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # ── Renderizado e Internos ─────────────────────────────────────────
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        self.state = None
        self.elapsed_steps = 0
        self.current_motor_cmd = 0.0
        self.last_action = 0.0  # Memoria para penalizar tirones bruscos

    def step(self, action: np.ndarray):
        assert self.state is not None, "Call reset() before step()"
        
        x, x_dot, theta, theta_dot = self.state
        self.elapsed_steps += 1

        # 1. Pipeline del motor y cálculo del tirón (delta_action)
        raw_cmd = np.clip(action[0], -1.0, 1.0)
        
        delta_action = raw_cmd - self.last_action
        self.last_action = raw_cmd  # Guardamos para el siguiente paso de simulación
        
        if abs(raw_cmd) < self.motor_deadzone:
            raw_cmd = 0.0

        # Filtro EMA para simular la inductancia/inercia real del motor
        self.current_motor_cmd = (self.motor_lag_alpha * self.current_motor_cmd) + ((1.0 - self.motor_lag_alpha) * raw_cmd)
        force = self.current_motor_cmd * self.force_mag

        # 2. Dinámica del péndulo invertido (MODELO DE CAJA RÍGIDA)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        
        # Denominador usando la Inercia Efectiva (I_eff) real de la caja
        denom = (self.I_eff / self.polemass_length) - (self.polemass_length * costheta**2 / self.total_mass)
        
        thetaacc = (self.gravity * sintheta - costheta * temp) / denom
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Fricción global básica (viscosa)
        xacc -= 0.1 * x_dot

        # 3. Integración Euler
        x += self.tau * x_dot
        x_dot += self.tau * xacc
        theta += self.tau * theta_dot
        theta_dot += self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # 4. Terminación
        terminated = bool(
            x < -self.x_threshold or x > self.x_threshold
            or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        )
        truncated = self.elapsed_steps >= self.max_episode_steps

        # 5. Recompensa Anti-Temblores (LQR-style + Smoothness Penalty)
        if not terminated:
            reward = 1.0 \
                     - 2.0 * (theta / self.theta_threshold_radians)**2 \
                     - 0.1 * (x / self.x_threshold)**2 \
                     - 0.1 * (theta_dot)**2 \
                     - 0.1 * (raw_cmd)**2 \
                     #- 0.8 * (delta_action)**2  # Castigo severo a los tirones bruscos
        else:
            reward = -10.0

        # 6. Observación con ruido (Simula IMU + Jitter de encoder)
        obs_noise = self.np_random.normal(0, [0.005, 0.02, 0.005, 0.05], size=4)
        obs = self.state + obs_noise

        if self.render_mode == "human":
            self.render()

        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.elapsed_steps = 0
        self.current_motor_cmd = 0.0
        self.last_action = 0.0  # Resetear la memoria de acción al iniciar el episodio

        # ── Domain Randomization (Físicas variadas por episodio) ───────────
        self.masscart = self.masscart_nom * self.np_random.uniform(0.85, 1.15)
        self.masspole = self.masspole_nom * self.np_random.uniform(0.85, 1.15)
        
        # Variamos ligeramente las dimensiones para que la red sea robusta
        box_h = self.box_height_nom * self.np_random.uniform(0.9, 1.1)
        box_d = self.box_depth_nom * self.np_random.uniform(0.9, 1.1)
        
        # Distancia al Centro de Masa (mitad de la altura total)
        self.length = box_h / 2.0
        
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length

        # CÁLCULO DE INERCIA DE LA CAJA
        # I_com = 1/12 * m * (h^2 + d^2)
        self.I_com = (1.0 / 12.0) * self.masspole * (box_h**2 + box_d**2)
        # Teorema de Steiner para mover el eje al motor: I_eff = I_com + m * l^2
        self.I_eff = self.I_com + self.masspole * (self.length**2)

        self.force_mag = 5.0 * self.np_random.uniform(0.80, 1.20)
        
        # Incrementamos ligeramente el lag simulado para forzar al agente a ser más precavido
        self.motor_lag_alpha = self.np_random.uniform(0.75, 0.95)

        # ── Inicialización Agresiva del estado ────────────────────────────
        # Forzamos al robot a empezar en situaciones de casi-caída
        # para que aprenda a recuperarse desde ángulos extremos.
        self.state = np.array([
            self.np_random.uniform(-0.1, 0.1),   # Posición X (metros)
            self.np_random.uniform(-0.2, 0.2),   # Velocidad X (m/s)
            self.np_random.uniform(-0.45, 0.45), # Inclinación inicial de hasta ±25º (en radianes)
            self.np_random.uniform(-0.8, 0.8)    # Velocidad angular (como si lo hubieran empujado)
        ], dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        obs_noise = self.np_random.normal(0, [0.005, 0.02, 0.005, 0.05], size=4)
        obs = self.state + obs_noise

        return np.array(obs, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled("pygame is required") from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("Bala2Fire — Digital Twin v2")
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

        COLOR_BG          = (245, 245, 248)
        COLOR_GROUND_FILL = (210, 210, 210)
        COLOR_GROUND_LINE = ( 60,  60,  60)
        COLOR_TYRE        = ( 35,  35,  35)
        COLOR_HUB         = ( 50,  50,  50)
        COLOR_SHADOW_WHEEL = (110, 110, 110)
        COLOR_STEM        = ( 75,  75,  75)
        COLOR_BODY        = (255, 120,   0)
        COLOR_LCD_DARK    = ( 10,  80, 160)
        COLOR_BTN         = (215, 215, 215)

        surf = pygame.Surface((self.screen_width, self.screen_height))
        surf.fill(COLOR_BG)

        def rot(local_x, local_y_up):
            vec = pygame.math.Vector2(local_x, local_y_up).rotate_rad(-theta)
            return (vec.x + pivot_x, vec.y + pivot_y)

        def quad(corners):
            return [rot(lx, ly) for lx, ly in corners]

        stem_w, stem_h = 10, 50
        box_w,  box_h  = 66, 32
        lcd_w,  lcd_y_bot = 38, stem_h + 5
        lcd_h   = 20
        btn_w,  btn_h,  btn_y_bot = 10, 6, stem_h + box_h - 3
        btn_xs  = [-20, 0, 20]

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
            btn_pts = quad([
                (bx_loc-btn_w/2, btn_y_bot-btn_h), (bx_loc-btn_w/2, btn_y_bot),
                (bx_loc+btn_w/2, btn_y_bot),       (bx_loc+btn_w/2, btn_y_bot-btn_h),
            ])
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
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False