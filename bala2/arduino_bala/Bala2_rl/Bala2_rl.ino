#define M5STACK_MPU6886 

#include <M5Stack.h>
#include "freertos/FreeRTOS.h"
#include "imu_filter.h"
#include "MadgwickAHRS.h"
#include "bala.h"
#include "pid.h"
#include "calibration.h"

// --- INTERFAZ CON TU RED NEURONAL (implementada en mi_modelo.cpp) ---
extern int16_t update_nn_motor_speed(float x, float x_dot, float theta, float theta_dot);
// -------------------------------------

// Umbral de normalización de x — DEBE coincidir con x_threshold del gym (2.4m).
static const float X_THRESHOLD = 2.4f;

// Escala de salida del motor — válvula de ajuste sin reentrenar.
// Si el robot sobrecompensa, bajar este valor (probar 0.55, 0.50...).
// Si subcompensa y no puede balancear, subirlo (0.70, 0.75...).
// Valor inicial: 0.65 (asume fuerza real ≈ 1.5x fuerza simulada).
// Razonamiento: force_mag=5N en sim, motores N20 reales pueden dar 7-10N.
static const float MOTOR_SCALE = 0.45f;

extern uint8_t bala_img[41056];
static void PIDTask(void *arg);
static void draw_waveform();

static float angle_point = -1.5;
bool calibration_mode = false;
bool is_shutting_down = false;

Bala bala;

void setup(){
  M5.begin(true, false, false, false);
  Serial.begin(115200);
  M5.IMU.Init();

  int16_t x_offset, y_offset, z_offset;
  float angle_center;
  calibrationInit();

  if (M5.BtnB.isPressed()) {
    calibrationGryo();
    calibration_mode = true;
  }

  calibrationGet(&x_offset, &y_offset, &z_offset, &angle_center);
  angle_point = angle_center;

  SemaphoreHandle_t i2c_mutex;
  i2c_mutex = xSemaphoreCreateMutex();
  bala.SetMutex(&i2c_mutex);   
  
  ImuTaskStart(x_offset, y_offset, z_offset, &i2c_mutex);
  xTaskCreatePinnedToCore(PIDTask, "pid_task", 8 * 1024, NULL, 4, NULL, 1);
  
  M5.Lcd.drawJpg(bala_img, 41056);
}

void loop() {
  static uint32_t next_show_time = 0;
  vTaskDelay(pdMS_TO_TICKS(5));
  
  if(millis() > next_show_time) {
    draw_waveform();
    next_show_time = millis() + 10;
  }

  M5.update();

  if (M5.BtnB.pressedFor(2000)) {
    is_shutting_down = true;
    bala.SetSpeed(0, 0);
    M5.Lcd.fillScreen(TFT_BLACK);
    M5.Lcd.setTextColor(TFT_WHITE);
    M5.Lcd.drawCentreString("Apagando RL Mode...", 160, 100, 2);
    delay(1000);
    M5.Power.powerOFF();
  }
} 

static void PIDTask(void *arg) {
  float x = 0, x_dot = 0;
  float theta = 0, theta_dot = 0;
  
  int32_t last_encoder = 0;
  uint32_t last_ticks = 0;
  const float dt = 0.005f;

  for(;;) {
    vTaskDelayUntil(&last_ticks, pdMS_TO_TICKS(5));

    if (is_shutting_down) {
      bala.SetSpeed(0, 0);
      continue;
    }

    // ── 1. THETA (ángulo en radianes) ──────────────────────────────────
    float current_theta_deg = getAngle() - angle_point;
    theta = current_theta_deg * (PI / 180.0f);
    
    // ── 2. THETA_DOT (velocidad angular rad/s) ─────────────────────────
    // M5.IMU.getGyroData() devuelve valores YA escalados y con el offset
    // interno del MPU6886 aplicado por el driver. NO hay que re-aplicar
    // el offset manualmente: hacerlo introduce un bias fijo permanente.
    // BUG ANTERIOR: calibrated_gyroX = gyroX + (gryo_x_offset * 0.061035f)
    //               duplicaba el offset que el driver ya aplicó.
    float gyroX, gyroY, gyroZ;
    M5.IMU.getGyroData(&gyroX, &gyroY, &gyroZ);
    theta_dot = gyroX * (PI / 180.0f);

    // ── 3. X (posición en metros) ──────────────────────────────────────
    bala.UpdateEncoder();
    int32_t current_encoder = (bala.wheel_left_encoder + bala.wheel_right_encoder) / 2;
    x = (float)current_encoder * 0.001f;

    // ── 4. X_DOT (velocidad lineal m/s, filtrada con IIR 0.7/0.3) ─────
    // El filtro debe ser IDÉNTICO al del gym (_xdot_filtered en _get_observation).
    // Gym:     self._xdot_filtered = 0.7 * self._xdot_filtered + 0.3 * instant_xdot
    // Arduino: x_dot = 0.7f * x_dot + 0.3f * instant_x_dot   ← igual
    float instant_x_dot = (x - (last_encoder * 0.001f)) / dt;
    x_dot = 0.7f * x_dot + 0.3f * instant_x_dot;
    last_encoder = current_encoder;

    // ── 5. NORMALIZACIÓN ANTES DE PASAR A LA RED ──────────────────────
    // CORRECCIÓN CRÍTICA: el gym entrena con x/X_THRESHOLD como primera
    // observación (ver _get_observation: x_obs_norm = x_obs / self.x_threshold).
    // Sin normalizar, a x=0.3m la red recibe 0.3 en lugar de 0.125 →
    // interpreta la posición como 2.4× más lejos → sobrecompensa gravemente.
    float x_norm = x / X_THRESHOLD;

    // ── 6. INFERENCIA Y ACTUACIÓN ─────────────────────────────────────
    if(fabs(current_theta_deg) < 70) {
      int16_t pwm_output = update_nn_motor_speed(x_norm, x_dot, theta, theta_dot);
      
      // Aplicar escala de fuerza — ajustar MOTOR_SCALE si sobrecompensa.
      // No aplicar antes de pasar a la red; la red sigue viendo observaciones reales.
      pwm_output = (int16_t)(pwm_output * MOTOR_SCALE);
      
      if(pwm_output > 1023) pwm_output = 1023;
      if(pwm_output < -1023) pwm_output = -1023;
      
      bala.SetSpeed(pwm_output, pwm_output);
    } else {
      bala.SetSpeed(0, 0);
      x = 0; x_dot = 0;
      bala.SetEncoder(0, 0);
    }
  }
}

static void draw_waveform() {
  #define MAX_LEN 120
  #define X_OFFSET 100
  #define Y_OFFSET 95
  #define X_SCALE 3
  static int16_t val_buf[MAX_LEN] = {0};
  static int16_t pt = MAX_LEN - 1;
  val_buf[pt] = constrain((int16_t)(getAngle() * X_SCALE), -50, 50);

  if (--pt < 0) {
    pt = MAX_LEN - 1;
  }

  for (int i = 1; i < (MAX_LEN); i++) {
    uint16_t now_pt = (pt + i) % (MAX_LEN);
    M5.Lcd.drawLine(i + X_OFFSET, val_buf[(now_pt + 1) % MAX_LEN] + Y_OFFSET, i + 1 + X_OFFSET, val_buf[(now_pt + 2) % MAX_LEN] + Y_OFFSET, TFT_BLACK);
    if (i < MAX_LEN - 1) {
      M5.Lcd.drawLine(i + X_OFFSET, val_buf[now_pt] + Y_OFFSET, i + 1 + X_OFFSET, val_buf[(now_pt + 1) % MAX_LEN] + Y_OFFSET, TFT_GREEN);
    }
  }
}
