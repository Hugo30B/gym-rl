import time
from bala2_env import Bala2Env

# 1. Instanciar el entorno con renderizado visual activado
env = Bala2Env(render_mode="human")

# 2. Vamos a probar 5 "intentos" (episodios)
episodios = 5

for ep in range(episodios):
    # Resetear el entorno para empezar desde el centro
    obs, info = env.reset()
    terminated = False
    puntuacion = 0
    pasos = 0

    print(f"\n--- Iniciando Episodio {ep + 1} ---")

    # Bucle que corre hasta que el robot se caiga
    while not terminated:
        # 3. Tomar una acción ALEATORIA (Un número al azar entre -1.0 y 1.0)
        # En el futuro, aquí usaremos: accion = modelo.predict(obs)
        accion_aleatoria = env.action_space.sample()

        # 4. Aplicar la acción (mandar el PWM a los motores virtuales)
        obs, reward, terminated, truncated, info = env.step(accion_aleatoria)
        
        puntuacion += reward
        pasos += 1

        # Pausa para que la ventana de PyGame no vaya a velocidad hiper-luz
        # 0.02 segundos = 50 FPS (Coincide con los 50Hz del Bala2Fire)
        time.sleep(0.02) 

    print(f"El robot sobrevivió {pasos} pasos. Puntuación final: {puntuacion:.2f}")

# 5. Cerrar la ventana limpiamente al terminar
env.close()
print("\n¡Prueba finalizada!")