#include <Arduino.h>
#include <math.h>
#include "model_data.h"

// Función de activación Tanh (estándar en PPO de Stable Baselines 3)
float activation(float x) {
    return tanhf(x);
}

// Función principal de inferencia
// Recibe los 4 valores del entorno Gym: [x, x_dot, theta, theta_dot]
// x: posición (metros)
// x_dot: velocidad lineal (m/s)
// theta: ángulo (radianes)
// theta_dot: velocidad angular (rad/s)
int16_t update_nn_motor_speed(float x, float x_dot, float theta, float theta_dot) {
    
    // 1. NORMALIZACIÓN (IMPORTANTE: Debe coincidir con tu entrenamiento en Gym)
    // Se ha ajustado el umbral de theta a 0.4189f (24 grados) para coincidir con el entorno de entrenamiento
    // y suavizar la respuesta del control.
    float input[4];
    input[0] = x / 2.4f;         // x_threshold
    input[1] = x_dot / 5.0f;     // Escala típica de velocidad
    input[2] = theta / 0.4189f;  // theta_threshold (±24 grados para normalizar mejor)
    input[3] = theta_dot / 10.0f; // Escala típica de velocidad angular

    // Capas intermedias
    float h1[32];
    float h2[32];

    // --- CAPA 1: Input (4) -> Hidden 1 (32) ---
    for (int i = 0; i < 32; i++) {
        h1[i] = layer1_b[i]; // Cargar bias
        for (int j = 0; j < 4; j++) {
            h1[i] += input[j] * layer1_w[i][j];
        }
        h1[i] = activation(h1[i]);
    }

    // --- CAPA 2: Hidden 1 (32) -> Hidden 2 (32) ---
    for (int i = 0; i < 32; i++) {
        h2[i] = layer2_b[i]; // Cargar bias
        for (int j = 0; j < 32; j++) {
            h2[i] += h1[j] * layer2_w[i][j];
        }
        h2[i] = activation(h2[i]);
    }

    // --- CAPA DE SALIDA: Hidden 2 (32) -> Action (1) ---
    float action = output_b[0];
    for (int j = 0; j < 32; j++) {
        action += h2[j] * output_w[0][j];
    }

    // La política de PPO suele usar Tanh para acotar la acción
    action = tanhf(action);

    // --- ESCALADO A PWM ---
    // Convertimos el rango [-1, 1] a [-1023, 1023]
    int16_t pwm = (int16_t)(action * 1023.0f);

    // Mantenemos la inversión de los motores solicitada por el usuario
    return pwm;
}
