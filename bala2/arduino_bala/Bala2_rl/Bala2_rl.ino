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

extern uint8_t bala_img[41056];
static void PIDTask(void *arg);
static void draw_waveform();

static float angle_point = -1.5;
bool calibration_mode = false;
bool is_shutting_down = false;

Bala bala;

void setup(){
  // Inicialización de M5Stack y sensores
  M5.begin(true, false, false, false);
  Serial.begin(115200);
  M5.IMU.Init();

  int16_t x_offset, y_offset, z_offset;
  float angle_center;
  calibrationInit();

  // Entrar en modo calibración si se mantiene pulsado BtnB al encender
  if (M5.BtnB.isPressed()) {
    calibrationGryo();
    calibration_mode = true;
  }

  // Cargar el punto de equilibrio guardado
  calibrationGet(&x_offset, &y_offset, &z_offset, &angle_center);
  angle_point = angle_center;

  SemaphoreHandle_t i2c_mutex;
  i2c_mutex = xSemaphoreCreateMutex();
  bala.SetMutex(&i2c_mutex);   
  
  // Iniciar tarea de la IMU (Filtro Madgwick independiente)
  ImuTaskStart(x_offset, y_offset, z_offset, &i2c_mutex);
  
  // Tarea de Control RL: 8KB de pila para la red neuronal
  xTaskCreatePinnedToCore(PIDTask, "pid_task", 8 * 1024, NULL, 4, NULL, 1);
  
  M5.Lcd.drawJpg(bala_img, 41056);
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(10));
  M5.update();

  // Función de apagado por seguridad (BtnB pulsado 2 segundos)
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
  // Variables de estado del entorno (CartPole Style)
  float x = 0, x_dot = 0;
  float theta = 0, theta_dot = 0;
  
  // Variables auxiliares
  int32_t last_encoder = 0;
  uint32_t last_ticks = 0;
  const float dt = 0.005f; // Bucle a 200Hz

  for(;;) {
    vTaskDelayUntil(&last_ticks, pdMS_TO_TICKS(5));

    if (is_shutting_down) {
      bala.SetSpeed(0, 0);
      continue;
    }

    // 1. OBTENER THETA (Ángulo en Radianes)
    // El filtro de Madgwick nos da el ángulo filtrado. Restamos el offset de calibración.
    float current_theta_deg = getAngle() - angle_point;
    theta = current_theta_deg * (PI / 180.0f);
    
    // 2. OBTENER THETA_DOT (Velocidad angular rad/s)
    // Usamos el giroscopio directamente (eje Y para el balanceo)
    float gyroX, gyroY, gyroZ;
    M5.IMU.getGyroData(&gyroX, &gyroY, &gyroZ);
    theta_dot = gyroY * (PI / 180.0f);

    // 3. OBTENER X (Posición en metros)
    bala.UpdateEncoder();
    // Promedio de ambos encoders para posición central
    int32_t current_encoder = (bala.wheel_left_encoder + bala.wheel_right_encoder) / 2;
    // Factor 0.001f: asumiendo 1mm por pulso (ajustar si es necesario)
    x = (float)current_encoder * 0.001f; 

    // 4. OBTENER X_DOT (Velocidad lineal m/s)
    // Derivada numérica con filtro paso bajo suave para reducir ruido de cuantización
    float instant_x_dot = (x - (last_encoder * 0.001f)) / dt;
    x_dot = 0.7f * x_dot + 0.3f * instant_x_dot;
    last_encoder = current_encoder;

    // --- LÓGICA DE CONTROL ---
    // Si la inclinación es excesiva (>70°), paramos motores por seguridad
    if(fabs(current_theta_deg) < 70) {
      
      // INFERENCIA RL
      int16_t pwm_output = update_nn_motor_speed(x, x_dot, theta, theta_dot);
      
      // Límites físicos del driver I2C
      if(pwm_output > 1023) pwm_output = 1023;
      if(pwm_output < -1023) pwm_output = -1023;
      
      bala.SetSpeed(pwm_output, pwm_output);
    } else {
      // Robot caído: apagado de seguridad y reset de posición
      bala.SetSpeed(0, 0);
      x = 0; x_dot = 0;
      bala.SetEncoder(0, 0);
    }
  }
}

// Visualización de la gráfica (opcional)
static void draw_waveform() {
  // Se puede implementar para ver el ángulo en tiempo real en la pantalla
}
