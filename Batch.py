import numpy as np
import matplotlib.pyplot as plt

# Caricamento dei dati dai file (salvati come 2 x n)
T_meas_full = np.loadtxt("C:/Users/angel/OneDrive/Desktop/Tmes.txt", delimiter="\t")
T_ext_full  = np.loadtxt("C:/Users/angel/OneDrive/Desktop/Test.txt", delimiter="\t")

# Estrai la seconda riga (in MATLAB: mes(2,:) e est(2,:))
T_meas = T_meas_full[1, :]  # Seconda riga di Tmes.txt
T_ext  = T_ext_full[1, :]   # Seconda riga di Test.txt

h = 1  # Passo temporale

# Calcolo delle differenze
# y = T_meas[1:] - T_meas[:-1]  => corrisponde a T(k+1)-T(k)
y = T_meas[1:] - T_meas[:-1]
# X = -h * (T_meas[:-1] - T_ext[:-1])
X = -h * (T_meas[:-1] - T_ext[:-1])

# Costruisci Y e Phi come vettori colonna
Y_subset = y.reshape(-1, 1)
Phi_subset = X.reshape(-1, 1)

# Stima di K tramite metodi diretti
# Metodo pseudoinversa
K_pseudo = np.linalg.pinv(Phi_subset) @ Y_subset
print(f'K stimato tramite pseudoinversa: {K_pseudo[0, 0]}')

# Metodo LS classico
K_ls = np.linalg.inv(Phi_subset.T @ Phi_subset) @ (Phi_subset.T @ Y_subset)
print(f'K stimato tramite LS: {K_ls[0, 0]}')

# Metodo classico con somme
K_classic = np.sum(Y_subset * Phi_subset) / np.sum(Phi_subset**2)
print(f'K stimato con formula classica: {K_classic}')

# Stima di K tramite Batch Gradient Descent
K_gd = 0  # Inizializzazione del parametro K
tol = 1e-6
alpha = 1e-6
max_iter = 1000
loss_history = []

for iter in range(max_iter):
    error = Y_subset - K_gd * Phi_subset
    grad = -Phi_subset.T @ error  # grad è scalare (ma restituito come 1x1 array)
    K_gd = K_gd - alpha * grad[0, 0]
    loss = 0.5 * np.sum(error**2)
    loss_history.append(loss)
    if abs(grad[0, 0]) < tol:
        print(f'Convergenza raggiunta dopo {iter+1} iterazioni.')
        break

print(f'K stimato tramite Batch Gradient Descent: {K_gd}')

# Simulazione della temperatura utilizzando K stimato (usiamo K_pseudo come in MATLAB)
T_sim = np.zeros_like(T_meas)
T_sim[0] = T_meas[0]  # Condizione iniziale

for k in range(len(T_meas) - 1):
    T_sim[k+1] = T_sim[k] - h * K_pseudo[0, 0] * (T_sim[k] - T_ext[k])

# Confronto tra T_meas (misurata) e T_sim (simulata)
plt.figure()
plt.plot(np.arange(len(T_meas)), T_meas, 'b', label='Temperatura misurata', linewidth=1.5)
plt.plot(np.arange(len(T_meas)), T_sim, 'r--', label='Temperatura simulata', linewidth=1.5)
plt.legend()
plt.xlabel('Tempo (secondi)')
plt.ylabel('Temperatura (degC)')
plt.title('Confronto tra temperatura misurata e simulata')
plt.show()

# Andamento della funzione di costo durante il Batch Gradient Descent
plt.figure()
plt.plot(np.arange(1, len(loss_history)+1), loss_history, 'k', linewidth=1.5)
plt.xlabel('Iterazioni')
plt.ylabel('Funzione di costo J(K)')
plt.title('Andamento della funzione di costo durante il Batch Gradient Descent')
plt.grid(True)
plt.show()
