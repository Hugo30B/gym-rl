#include <Arduino.h>
#include <math.h>
#include "model_data.h"

// Función de activación Tanh (correcta SOLO para capas ocultas)
float activation(float x) {
    return tanhf(x);
}

int16_t update_nn_motor_speed(float x, float x_dot, float theta, float theta_dot) {
    

    float input[4] = {x, x_dot, theta, theta_dot};

    // Capas intermedias (Iniciadas a 0 por seguridad)
    float h1[32] = {0.0f};
    float h2[32] = {0.0f};

    // --- CAPA 1: Input (4) -> Hidden 1 (32) ---
    for (int i = 0; i < 32; i++) {
        h1[i] = layer1_b[i]; 
        for (int j = 0; j < 4; j++) {
            // En PyTorch los pesos son [out_features][in_features]
            // layer1_w debe ser un array [32][4]
            h1[i] += input[j] * layer1_w[i][j];
        }
        h1[i] = activation(h1[i]);
    }

    // --- CAPA 2: Hidden 1 (32) -> Hidden 2 (32) ---
    for (int i = 0; i < 32; i++) {
        h2[i] = layer2_b[i]; 
        for (int j = 0; j < 32; j++) {
            // layer2_w debe ser un array [32][32]
            h2[i] += h1[j] * layer2_w[i][j];
        }
        h2[i] = activation(h2[i]);
    }

    // --- CAPA DE SALIDA: Hidden 2 (32) -> Action (1) ---
    float action = output_b[0];
    for (int j = 0; j < 32; j++) {
        // output_w debe ser un array [1][32]
        action += h2[j] * output_w[0][j];
    }

    // Recortamos la acción bruta al rango [-1.0, 1.0] tal y como hace Gym
    if (action > 1.0f) action = 1.0f;
    if (action < -1.0f) action = -1.0f;

    // --- ESCALADO A PWM ---
    // Convertimos el rango [-1.0, 1.0] a [-1023, 1023]
    int16_t pwm = (int16_t)(action * 1023.0f);

    return pwm;
}