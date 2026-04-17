import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from collections import deque


class Bala2Env(gym.Env):
    """
    ## Description
    Digital twin of the M5Stack Bala2Fire — v2, Enhanced Sim-to-Real.

    Mejoras sobre v1:
    ─────────────────────────────────────────────────────────────────────────
    1. RECOMPENSA  → Exponencial centrada en θ=0. Sin cap explotable.
                     El agente DEBE balancear bien para maximizar.
    2. RUIDO SENSOR→ Por sensor: IMU (θ, θ̇) con drift de bias gaussiano.
                     Encoder de rueda (x, ẋ) con cuantización discreta.
    3. MOTOR       → Zona muerta realista + filtro de 1er orden (lag).
                     Ruido de torque aplicado DESPUÉS del lag, no antes.
    4. FRICCIÓN    → Coulomb (estática) + viscosa, randomizables por separado.
    5. DR extendida→ Incluye ganancia del motor y ambos coeficientes de
                     fricción, además de las masas y longitud.
    ─────────────────────────────────────────────────────────────────────────

    ## Observation Space
        [x, x_dot, theta, theta_dot]  (con ruido realista por canal)

    ## Action Space
        [-1.0, 1.0]  (comando normalizado al motor)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,   # Display a 50Hz — la simulación corre a 200Hz
    }

    def __init__(self, render_mode: str | None = None):

        # ── Constantes físicas nominales ───────────────────────────────────
        self.gravity        = 9.8
        self.masscart_nom   = 0.15    # kg  (chasis + ruedas)
        self.masspole_nom   = 0.25    # kg  (cuerpo superior)
        self.length_nom     = 0.06    # m   (dist. eje rueda → CoM cuerpo)
        self.force_mag      = 5.0     # N   (fuerza máxima aplicable)
        self.tau            = 0.005   # s   (200 Hz — igual que el real)

        # ── Modelo de fricción (Coulomb + viscosa) ─────────────────────────
        # f_total = f_coulomb * sign(ẋ) + f_viscous * ẋ
        # El BALA2 tiene gomas de silicona → Coulomb notable + viscosa baja.
        self.friction_coulomb_nom = 0.06   # N  (fricción estática/dinámica)
        self.friction_viscous_nom = 0.03   # N·s/m (amortiguamiento viscoso)

        # ── Modelo de motor ────────────────────────────────────────────────
        # Zona muerta: comandos |u| < deadzone → 0 N (umbral real ~4-6%)
        self.motor_deadzone   = 0.05
        # Lag de 1er orden: simula la inercia eléctrica y mecánica del motor
        # out[k] = α·out[k-1] + (1-α)·cmd[k]
        # τ_motor ≈ 56ms → α = exp(-dt/τ) = exp(-0.005/0.056) ≈ 0.915  @200Hz
        self.motor_lag_alpha  = 0.915
        # Ganancia nominal del motor (randomizable ±15%)
        self.motor_gain_nom   = 1.0
        # Ruido de torque (std relativo a la fuerza máxima): ±3%
        # Se aplica DESPUÉS del lag, simulando variación de voltaje/slip.
        self.torque_noise_std = 0.03

        # ── Ruido de sensores ──────────────────────────────────────────────
        # IMU (MPU6886 en el M5Stack): datos a 100 Hz, filtrado a 50 Hz.
        #   · Ruido gaussiano en ángulo (salida del complementary filter).
        #   · Ruido gaussiano en velocidad angular (giróscopo crudo).
        #   · Drift de bias: paseo aleatorio lento acotado (errores de integración).
        self.theta_noise_std      = 0.003   # rad  (~0.17°, típico tras filtrado)
        self.thetadot_noise_std   = 0.010   # rad/s (ruido del giróscopo)
        self.imu_bias_drift_std   = 0.000125  # rad/step (=0.025 rad/s, misma tasa que v1 @50Hz)
        self.imu_bias_limit       = 0.015   # rad   (bias máximo acotado)

        # Encoder de rueda (pulsos discretos):
        #   · Cuantización: resolución ~2mm por pulso del encoder.
        #   · Ruido gaussiano adicional por jitter del conteo.
        self.encoder_resolution   = 0.002   # m  (resolución de cuantización)
        self.x_noise_std          = 0.001   # m  (jitter gaussiano)
        self.xdot_noise_std       = 0.005   # m/s (derivada numérica, más ruidosa)

        # ── Latencia de control ────────────────────────────────────────────
        # 4 pasos × 5ms = 20ms (latencia típica del bucle BLE/I2C del BALA2)
        self.latency_steps  = 4
        self.action_queue   = deque(maxlen=self.latency_steps + 1)

        # ── Límites de episodio ────────────────────────────────────────────
        self.theta_threshold_radians = 12 * 2 * math.pi / 360   # ±12°
        self.x_threshold             = 2.4                       # ±2.4 m
        self.max_episode_steps       = 2000  # 2000 pasos × 5ms = 10s de episodio

        # ── Espacios de acción/observación ────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        obs_high = np.array(
            [self.x_threshold * 2, np.inf, self.theta_threshold_radians * 2, np.inf],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # ── Renderizado ────────────────────────────────────────────────────
        self.render_mode   = render_mode
        self.screen_width  = 600
        self.screen_height = 400
        self.screen        = None
        self.clock         = None
        self.isopen        = True

        # ── Estado interno ─────────────────────────────────────────────────
        self.state: np.ndarray | None = None
        self.steps_beyond_terminated: int | None = None
        self.elapsed_steps  = 0
        self._motor_output  = 0.0   # Estado del filtro de lag del motor
        self._imu_bias      = 0.0   # Bias acumulado del IMU

    # ══════════════════════════════════════════════════════════════════════
    # MODELO DE MOTOR
    # ══════════════════════════════════════════════════════════════════════

    def _apply_motor_model(self, raw_cmd: float) -> float:
        """
        Pipeline realista del actuador:

        1. Zona muerta — comandos pequeños no superan la fricción estática
           del tren de tracción. La escala posterior es lineal para que
           ±1.0 siga siendo el comando máximo.

        2. Filtro de 1er orden — la inductancia del motor y la inercia
           mecánica impiden cambios instantáneos de fuerza.

        3. Ganancia del motor — variación por tensión de batería, temperatura,
           etc. (domain randomization).

        4. Ruido de torque — fluctuaciones de corriente, slip de rueda,
           superficie irregular. Se clipa para no exceder force_mag.

        Devuelve la fuerza resultante en Newtons.
        """
        # 1. Zona muerta con re-escala lineal
        dz = self.motor_deadzone
        if abs(raw_cmd) <= dz:
            cmd_effective = 0.0
        else:
            cmd_effective = np.sign(raw_cmd) * (abs(raw_cmd) - dz) / (1.0 - dz)

        # 2. Filtro lag de 1er orden
        self._motor_output = (
            self.motor_lag_alpha * self._motor_output
            + (1.0 - self.motor_lag_alpha) * cmd_effective
        )

        # 3. Ganancia + ruido de torque (relativo a force_mag)
        noise = self.np_random.normal(0.0, self.torque_noise_std)
        force_normalized = np.clip(self._motor_output * self.motor_gain + noise, -1.0, 1.0)

        return float(force_normalized * self.force_mag)

    # ══════════════════════════════════════════════════════════════════════
    # MODELO DE FRICCIÓN
    # ══════════════════════════════════════════════════════════════════════

    def _friction_force(self, x_dot: float, xacc_ideal: float) -> float:
        """
        Fricción Coulomb + viscosa.

        La fricción de Coulomb se aplica como límite: si el robot está
        prácticamente parado (|ẋ| < ε), la fricción estática amortigua la
        aceleración en lugar de invertir el signo artificialmente.

        Retorna la fuerza de fricción (en N) a restar de xacc*total_mass.
        """
        eps = 0.005  # m/s  (umbral de velocidad casi nula)
        f_viscous = self.friction_viscous * x_dot

        if abs(x_dot) < eps:
            # Fricción estática: limita la aceleración, no la invierte
            f_coulomb = np.clip(-xacc_ideal * self.total_mass,
                                -self.friction_coulomb,
                                 self.friction_coulomb)
        else:
            f_coulomb = self.friction_coulomb * np.sign(x_dot)

        return f_viscous + f_coulomb

    # ══════════════════════════════════════════════════════════════════════
    # MODELO DE SENSORES
    # ══════════════════════════════════════════════════════════════════════

    def _get_observation(self) -> np.ndarray:
        """
        Observación con ruido realista por canal:

        · IMU (θ, θ̇):
            - Ruido gaussiano estacionario (filtrado complementario).
            - Drift de bias lento (paseo aleatorio acotado): el agente
              nunca ve el θ real exacto, igual que en el hardware.

        · Encoder (x, ẋ):
            - Cuantización discreta (resolución de pulso del encoder).
            - Ruido gaussiano adicional por jitter de conteo.
        """
        x, x_dot, theta, theta_dot = self.state

        # ── IMU ───────────────────────────────────────────────────────────
        # Drift del bias: paseo aleatorio acotado
        self._imu_bias += self.np_random.normal(0.0, self.imu_bias_drift_std)
        self._imu_bias = float(np.clip(self._imu_bias,
                                       -self.imu_bias_limit,
                                        self.imu_bias_limit))

        theta_obs    = theta     + self._imu_bias + self.np_random.normal(0.0, self.theta_noise_std)
        thetadot_obs = theta_dot + self.np_random.normal(0.0, self.thetadot_noise_std)

        # ── Encoder ───────────────────────────────────────────────────────
        x_obs    = x    + self.np_random.normal(0.0, self.x_noise_std)
        x_obs    = np.round(x_obs / self.encoder_resolution) * self.encoder_resolution
        xdot_obs = x_dot + self.np_random.normal(0.0, self.xdot_noise_std)

        return np.array([x_obs, xdot_obs, theta_obs, thetadot_obs], dtype=np.float32)

    # ══════════════════════════════════════════════════════════════════════
    # FUNCIÓN DE RECOMPENSA
    # ══════════════════════════════════════════════════════════════════════

    def _compute_reward(
        self,
        x: float,
        x_dot: float,
        theta: float,
        theta_dot: float,
        force: float,
    ) -> float:
        """
        Diseño de recompensa para sim-to-real:

        ┌─────────────────────────────────────────────────────────────────┐
        │  r_θ  = exp(-k_θ · θ²)          Pico 1.0 en θ=0               │
        │                                  ≈0.70 a ±5°  (señal rica)     │
        │                                  ≈0.10 a ±10° (umbral claro)   │
        │                                                                 │
        │  r_x  = exp(-k_x · x²)          Penaliza drift suave           │
        │                                  ≈0.78 a ±0.8m                 │
        │                                                                 │
        │  r_vel = -α_ẋ·ẋ² - α_θ̇·θ̇²      Amortiguamiento de velocidades │
        │                                  Favorece control suave         │
        │                                                                 │
        │  r_act = -α_u·(F/Fmax)²          Coste cuadrático de acción     │
        │                                  Evita saturar el motor         │
        │                                                                 │
        │  TOTAL = r_θ · r_x + r_vel + r_act                             │
        │                                                                 │
        │  Rango aprox: [-0.15, +1.0] por paso cuando el robot balancea. │
        │  Sin cap explotable: el agente DEBE mantener θ≈0 y x≈0.        │
        └─────────────────────────────────────────────────────────────────┘

        Por qué no hay cap:
          · r_θ · r_x nunca supera 1.0 por construcción matemática.
          · r_vel y r_act son negativos → no se puede "inflar" la recompensa
            quedándose quieto ni saturando el motor.
          · No hay término lineal constante que el agente pueda acumular
            sin hacer nada útil.
        """
        # Componente de ángulo — término principal
        k_theta = 15.0
        r_theta = float(np.exp(-k_theta * theta ** 2))

        # Componente de posición — penalización suave al alejarse del centro
        k_x = 0.30
        r_x = float(np.exp(-k_x * x ** 2))

        # Penalización de velocidades — fomenta estabilidad y suavidad
        alpha_xdot    = 0.015
        alpha_thetadot = 0.008
        r_vel = -alpha_xdot * x_dot ** 2 - alpha_thetadot * theta_dot ** 2

        # Coste cuadrático de acción — penaliza uso ineficiente del motor
        alpha_action = 0.04
        r_act = -alpha_action * (force / self.force_mag) ** 2

        return r_theta * r_x + r_vel + r_act

    # ══════════════════════════════════════════════════════════════════════
    # STEP
    # ══════════════════════════════════════════════════════════════════════

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action), f"{action!r} invalid"
        assert self.state is not None, "Call reset() before step()."

        self.elapsed_steps += 1
        x, x_dot, theta, theta_dot = self.state

        # ── 1. Latencia de control ─────────────────────────────────────────
        self.action_queue.append(action)
        delayed_action = (
            self.action_queue[0]
            if len(self.action_queue) > self.latency_steps
            else np.zeros(1, dtype=np.float32)
        )
        raw = float(np.clip(delayed_action, -1.0, 1.0)[0])

        # ── 2. Modelo de motor ─────────────────────────────────────────────
        force = self._apply_motor_model(raw)

        # ── 3. Dinámica del péndulo invertido ──────────────────────────────
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc_ideal = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # ── 4. Fricción Coulomb + viscosa ──────────────────────────────────
        f_friction = self._friction_force(x_dot, xacc_ideal)
        xacc = xacc_ideal - f_friction / self.total_mass

        # ── 5. Integración Euler ───────────────────────────────────────────
        x         = x         + self.tau * x_dot
        x_dot     = x_dot     + self.tau * xacc
        theta     = theta     + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)

        # ── 6. Terminación ─────────────────────────────────────────────────
        terminated = bool(
            x     < -self.x_threshold or x     > self.x_threshold
            or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        )
        truncated = self.elapsed_steps >= self.max_episode_steps

        # ── 7. Recompensa ──────────────────────────────────────────────────
        if not terminated:
            reward = self._compute_reward(x, x_dot, theta, theta_dot, force)
        else:
            # Señal negativa clara de fallo; mayor que cualquier step reward
            reward = -5.0

        # ── 8. Observación con ruido ───────────────────────────────────────
        obs = self._get_observation()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    # ══════════════════════════════════════════════════════════════════════
    # RESET + DOMAIN RANDOMIZATION
    # ══════════════════════════════════════════════════════════════════════

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.elapsed_steps  = 0
        self._motor_output  = 0.0
        self._imu_bias      = 0.0

        # ── Domain Randomization ───────────────────────────────────────────
        # Cada episodio: el robot real tiene variaciones por temperatura,
        # superficie, carga de batería, etc.
        rng = self.np_random

        # Masas y geometría (±12%)
        self.masscart = self.masscart_nom * rng.uniform(0.88, 1.12)
        self.masspole = self.masspole_nom * rng.uniform(0.88, 1.12)
        self.length   = self.length_nom   * rng.uniform(0.88, 1.12)

        # Fricción: Coulomb y viscosa independientes (±25%)
        # Rango amplio porque la superficie real varía mucho (suelo liso, alfombra…)
        self.friction_coulomb = self.friction_coulomb_nom * rng.uniform(0.75, 1.25)
        self.friction_viscous = self.friction_viscous_nom * rng.uniform(0.75, 1.25)

        # Ganancia del motor (±15%) — simula variación de voltaje de batería
        self.motor_gain = self.motor_gain_nom * rng.uniform(0.85, 1.15)

        # Constantes derivadas
        self.total_mass      = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length

        # Limpiar cola de latencia
        self.action_queue.clear()

        # Estado inicial pequeño (cerca del equilibrio)
        low, high = utils.maybe_parse_reset_bounds(options, -0.05, 0.05)
        self.state = rng.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), {}

    # ══════════════════════════════════════════════════════════════════════
    # RENDER (sin cambios respecto a v1)
    # ══════════════════════════════════════════════════════════════════════

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