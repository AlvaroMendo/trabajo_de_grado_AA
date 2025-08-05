# Análisis Modal Espectral
# Pórtico Plano de Concreto Reforzado de 5 niveles
# Autores:      Alejandra Sepúlveda
#               Alvaro Mendoza
# Supervisor:   John Ardila

# Unidades
# Longitud: m
# Masa:     kg
# Tiempo:   s
# Fuerza:   N = kg.m/s^2
# Esfuerzo: Pa = N/m^2 (módulo de elasticidad)

# Importar librerías
import openseespy.opensees as ops
import opsvis as opsv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

# Parámetros a modificar:
kEcfc = 4700 # relación Ec/(f'c^0.5)
# Número de pisos
Npisos = 10
# Sección de columnas y vigas [b, h] en cm
scol = [30.0, 30.0]
svig = [30.0, 50.0]
# Considerar carga viva: 0-No y 1-Si
consL = 0 
# Amenaza sísmica
Aa, Av, I, TS = 0.15, 0.15, 1.0, 'C'
# FA: Factor de Amplificación para (0.8Vs)/Vst > 1.0
FA = 1.0
# Modos
Nmodes = 3 # número de modos de vibración a calcular
Nmode = 3 # número de modos de vibración a graficar

# Datos de entrada
g = 9.81 # acc. gravedad (m/s^2)

nVx, nPy = 3, Npisos # número de vanos y pisos
LVx, LVz, dHy = 5.0, 5.0, 3.0 # longitud de vanos en x, z y altura en y
nnod = (nVx+1)*(nPy+1)
ndpi = (nVx+1) # número de nodos por piso
ncol = ndpi*nPy # número de columnas
nvig = nVx*nPy # número de vigas
nele = ncol+nvig # número de elementos (columnas y vigas)
nodos = np.zeros((nnod,2),dtype = np.float64)
elementos = np.zeros((nele,2),dtype = np.int64)
cont = 0
dLy = 0.0
for j in range(nPy+1):
    dLx = 0.0
    for k in range(nVx+1):
        nodos[cont,:] = [dLx, dLy]
        cont+=1
        dLx = dLx+LVx
    dLy = dLy+dHy
cont = 0
for i in range(ncol):
    elementos[i,:] = [i, i+ndpi]
    cont+=1
contx = ndpi
conty = ndpi+1
for i in range(nPy):
    for j in range(nVx):
        elementos[cont,:] = [contx, conty]
        cont+=1
        contx+=1
        conty = contx+1
    contx+=1
    conty = conty+1

# Materiales:
# Concreto
fc = 21e6 # resistencia especificada a la compresión (Pa)
Ec = kEcfc*(fc*1e-6)**0.5*1e6 # módulo de elasticidad (Pa)
vc = 0.2 # coeficiente de Poisson
Gc = Ec/(2*(1+vc)) # módulo de corte (Pa)
fct = 0.62*(fc*1e-6)**0.5*1e6 # resistencia a la tracción
gammac = 24e3 # peso específico (N/m^3)
# Acero
fy = 420e6 # esfuerzo de fluencia (Pa)
Es = 200e9 # módulo de elasticidad (Pa)

# Columnas
bc, hc = scol[0]*1e-2, scol[1]*1e-2
Acc , I1c = bc*hc, 1/12*bc*hc**3
mc = Acc*gammac/g

# Secciones transversales:
# Vigas
bv, hv = svig[0]*1e-2, svig[1]*1e-2
Avc , I1v = bv*hv, 1/12*bv*hv**3
mv = Avc*gammac/g

# Cargas: Muerta y Viva
wle = 2.5e3 # carga de entrepiso (N/m^2)
wfp = 3.00e3 # carga fachadas y particiones (N/m^2)
wap = 1.60e3 # carga acabados y afinado de piso (N/m^2)
wL = 1.80e3 # carga viva (N/m^2)
wT = 1.00*(wle+wfp+wap)+0.25*wL*consL

# Definición del modelo
ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 3)

# Definir los nodos
for i, nodoi in enumerate(nodos):
    ops.node(i+1, *nodoi)

# Apoyos: restricciones en el nivel y = 0.0 m
ops.fixY(0.0, *[1, 1, 1])

# Definición de elementos
CTransform = 1
BTransform = 2

ops.geomTransf('PDelta', CTransform, *[1, 0, 0])
ops.geomTransf('Linear', BTransform, *[0, -1, 0])

# Ensamble de columnas y vigas
for i, elem in enumerate(elementos):
    nodoi = int(elem[0]+1) # nodo inicial
    nodoj = int(elem[1]+1) # nodo final
    if i <= ncol-1:
        # Columnas
        ops.element('elasticBeamColumn', i+1, nodoi, nodoj,
                    Acc, Ec, I1c, CTransform, '-mass', mc)
    else:
        # Vigas
        ops.element('elasticBeamColumn', i+1, nodoi, nodoj,
                    Avc, Ec, I1v, BTransform, '-mass', mv)

# Graficar nodos y elementos
figsz = (nVx*LVx, nPy*dHy)
fmt_model = {'color': 'blue', 'linestyle': 'solid', 'linewidth': 1.6, 'marker': '.', 'markersize': 8}
opsv.plot_model(node_labels=1, element_labels=1, fmt_model = fmt_model, fig_wi_he = figsz)
plt.savefig('NodElemP2D.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Definición de masa concentrada en nodos
# CM y masa/nodo para piso tipo
CM = nVx*LVx/2
mnd = wT*LVx*LVz/g
# CM y masa/nodo para techo
CMt = nVx*LVx/2
mndt = (wT-wfp)*LVx*LVz/g

# Agregar nodos CM en el modelo
# GDL de cada entrepiso
gdlcm = [0, 1, 1]

# Crear nodos de CM para cada nivel
Hy = np.ones(nPy)*dHy # alturas de entrepiso de cada nivel
for i in range(nPy):
    Tagnode_CM = len(nodos)+i+1 # etiqueta de los nodos de CM
    if i != nPy-1:
        ops.node(Tagnode_CM, CM, sum(Hy[:i+1]))
    else:
        ops.node(Tagnode_CM, CMt, sum(Hy[:i+1]))
    
    ops.fix(Tagnode_CM, *gdlcm) # fijar restricciones

# Definir los diafragmas de piso (constraints)
perpDirn = 1
for i in range(nPy):
    Tagnode_CM = len(nodos)+i+1 # etiqueta de los nodos de CM
    nodoi = (ndpi+1)+i*ndpi # nodo incial cada nivel
    nodof = nodoi + ndpi # nodo final de cada nivel
    for cNodeTags in range(nodoi, nodof):
        ops.rigidDiaphragm(perpDirn, Tagnode_CM, cNodeTags)

# Asignar masas a cada nivel
for i in range(nPy):
    nodoi = (ndpi+1)+i*ndpi # nodo incial cada nivel
    for j in range(ndpi):
        if i != nPy-1:
            if j == 0 or j == ndpi-1:
                ops.mass(nodoi+j, *[mnd/2, mnd/2, 0.0]) # asignando las masa
            else:
                ops.mass(nodoi+j, *[mnd, mnd, 0.0]) # asignando las masa
        else:
            if j == 0 or j == ndpi-1:
                ops.mass(nodoi+j, *[mndt/2, mndt/2, 0.0])
            else:
                ops.mass(nodoi+j, *[mndt, mndt, 0.0])

opsv.plot_model()

# Eigevalores y Eigenvectores

vals = np.array(ops.eigen('-fullGenLapack', Nmodes)) # eigenvalores
omega = np.sqrt(vals) # frecuencia angular (rad/s)
Tmodes = 2.0*np.pi/omega # periodo (s)
frec = 1.0/Tmodes # frecuencia (Hz o s^-1)

# tabular resultados
df = pd.DataFrame(columns = ['Modo', 'w (rad/s)', 'f (Hz)', 'T (s)'])
for i in range(Nmodes):
    df = df._append({'Modo': i+1, 'w (rad/s)': omega[i], 'f (Hz)': frec[i], 'T (s)': Tmodes[i]}, ignore_index = True)

df = df.astype({'Modo': int})
display(df.round(2))

# Graficar modos de vibración
fmt_undefo = {'color': 'gray', 'linestyle': (0,(0.7,1.5)), 'linewidth': 1.8,
              'marker': '', 'markersize': 1.0}
for i in range(Nmode):
    opsv.plot_mode_shape(i+1,endDispFlag = 0, fig_wi_he = figsz, 
                          fmt_undefo = fmt_undefo, 
                          node_supports = False,az_el = (-140,30))
    plt.title(f'T[{i+1}]: {Tmodes[i]: .2f} s')
    plt.savefig(f'FormaModal_{i+1}.png', dpi = 300, bbox_inches = 'tight')

# Análsisis Modal Espectral

# Obtener la matriz de masas
ops.wipeAnalysis()
ops.system('FullGeneral')
ops.numberer('Plain')
ops.constraints('Transformation')
ops.algorithm('Linear')
ops.analysis('Transient')
ops.integrator('GimmeMCK', 1.0, 0.0, 0.0)
ops.analyze(1, 0)

# Matriz de masa (Ms)
NGDLT = ops.systemSize() # GDL = 3gdl*Npisos*(ndpi+1CM)
Mmatriz = ops.printA('-ret')
Mmatriz = np.array(Mmatriz)
Mmatriz.shape = (NGDLT, NGDLT)
Ms = Mmatriz[-nPy:, -nPy:]

# Obtener los modos de vibración
Tags = ops.getNodeTags() # para obtener etiqueta de nodos = ndpi*(Npisos+1)+NCM
# print(Tags)

# Formas Modales (3GDL/piso)
modo = np.zeros((Nmodes, nPy))
for j in range(1, Nmodes+1):
    ind = 0
    for i in Tags[-nPy:]:
        temp = ops.nodeEigenvector(i, j)
        modo[j-1, ind] = temp[0]
        ind+=1

# Definir valores iniciales
ux = np.ones(nPy)
sumux = 0.0
ni = 0
Mux = sum(sum(Ms)) # masa traslacional en x (kg)

# Tabular resultados: contribución de masa de los modos
df1 = pd.DataFrame(columns = ['Modo', 'T (s)', 'sum_ux'])
for j in range(1,Nmodes+1):
    FPux = modo[j-1].T@ Ms @ ux
    FPRux = FPux**2/Mux
    sumux = sumux+FPRux
    if sumux >= 0.90 and ni == 0:
        ni = j
    df1 = df1._append({'Modo': j, 'T (s)': Tmodes[j-1], 'sum_ux': sumux}, ignore_index = True)

df1 = df1.astype({'Modo': int})
display(df1.round(3))
print(f'Cantidad de modos requeridos: {ni}')
nmreq = ni

# Análisis Modal Espectral: Superposión Modal
def AnalisisModal(Aa, Av, I, TS, Ms, modo, Tmodes, NGDL, ni, ux, FA):
    g = 9.81
    # Valores iniciales:
    D_RCSCx, Δ_RCSCx, V_RCSCx = np.zeros(NGDL), np.zeros(NGDL), np.zeros(NGDL)
    
    for j in range(1, ni+1):
        FPux = modo[j-1].T@ Ms @ ux
        
        Sa = SaNSR10(Tmodes[j-1], Aa, Av, I, TS)*g*FA
        Sd = (Tmodes[j-1]/(2*np.pi))**2 * Sa
        
        # Respuesta en x
        respDx = Sd*FPux*modo[j-1]
        respAx = Sa*FPux*Ms @ modo[j-1]
        D_RCSCx = D_RCSCx + (respDx)**2
        respDx[1:] = respDx[1:] - respDx[:-1]
        Δ_RCSCx = Δ_RCSCx + respDx**2
        V_RCSCx = V_RCSCx + (np.cumsum(respAx[::-1])[::-1])**2
        
    # Obtener la respuesta
    Dx, Δx, Vx = D_RCSCx**0.5, Δ_RCSCx**0.5, V_RCSCx**0.5
    
    # Tabular resultados: V (kN), D (cm)
    df = pd.DataFrame(columns = ['Nivel', 'Vx (kN)', 'Dx (cm)'])
    for i in range(int(NGDL)):
        df = df._append({'Nivel': i+1, 'Vx (kN)': Vx[i]*1e-3,
                          'Dx (cm)': Dx[i]*1e2}, ignore_index = True)
    return Dx, Δx, Vx, df


# Espectro elástico de Diseño (NSR-10)
def SaNSR10(T, Aa, Av, I, TS):
    # Definición de los tipos de suelo (Letra: Número)
    TS_tx2num = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    
    # Convertir el tipo de suelo TS de letra a número
    Ts_num = TS_tx2num.get(TS)
    
    if Ts_num is None:
        raise ValueError('Tipo de suelo no encontrado.')
    
    # Definición de los factores Fa y Fv para cada tipo de suelo
    factores_de_suelo = {
        1: (0.8, 0.8),
        2: (1, 1),
        3: (
            1.2 if Aa <= 0.2 else (1.0 - 1.2) / (0.4 - 0.2) * (Aa - 0.2) + 1.2 if Aa <= 0.1 else 1,
            1.7 if Av <= 0.1 else (1.3 - 1.7) / (0.5 - 0.1) * (Av - 0.1) + 1.7 if Av <= 0.5 else 1.3
            ),
        4: (
            1.6 if Aa <= 0.1 else (
                (1.2 - 1.6) / (0.3 - 0.1) * (Aa - 0.1) + 1.6 if Aa <= 0.3 else (1.0 - 1.2) / (0.5 - 0.3) * (Aa - 0.3) + 1.2 if Av <= 0.5 else 1.0
                                  ),
            2.4 if Av <= 0.1 else (
                (2 - 2.4) / (0.2 - 0.1) * (Av - 0.1) + 2.4 if Av <= 0.2 else (
                    (1.6 - 2) / (0.4 - 0.2) * (Av - 0.2) + 2 if Av <= 0.4 else (1.5 - 1.6) / (0.5 - 0.4) * (Av - 0.4) + 1.6 if Av <= 0.5 else 1.5
                                                                              )
                                  ),
            ),
        5: (
            2.5 if Aa <= 0.1 else (
                (1.7 - 2.5) / (0.2 - 0.1) * (Aa - 0.1) + 2.5 if Aa <= 0.2 else (
                    (1.2 - 1.7) / (0.3 - 0.2) * (Aa - 0.2) + 1.7 if Aa <= 0.3 else (0.9 - 1.2) / (0.4 - 0.3) * (Aa - 0.3) + 1.2 if Aa <= 0.4 else 0.9
                                                                                )
                                  ),
            3.5 if Av <= 0.1 else (
                (3.2 - 3.5) / (0.2 - 0.1) * (Av - 0.1) + 3.5 if Av <= 0.2 else (
                    (2.4 - 3.2) / (0.4 - 0.2) * (Av - 0.2) + 3.2 if Av <= 0.4 else 2.4
                                                                                )
                                  ),
            )
                        }

    # Obtener los factores correspondientes al tipo de suelo
    Fa, Fv = factores_de_suelo[Ts_num]
    
    # Calcular los periodos
    T0 = 0.1 * (Av * Fv) / (Aa * Fa)
    TC = 0.48 * (Av * Fv) / (Aa * Fa)
    TL = 2.4 * Fv

    # Determinar la respuesta espectral según el periodo
    if T <= T0:
        return 2.5 * Aa * Fa * I
    elif T <= TC:
        return 2.5 * Aa * Fa * I
    elif T <= TL:
        return 1.2 * Av * Fv * I / T
    else:
        return 1.2 * Av * Fv * TL * I / (T ** 2)

# Graficar el espectro de diseño
Tnf = 5.0 # periodo final (s) para graficar
dTn = 0.01 # paso del Periodo (s)
Tn = np.arange(0.0, Tnf+0.01, dTn)
San = [SaNSR10(T, Aa, Av, I, TS) for T in Tn]

plt.figure(figsize = (7, 4))
plt.plot(Tn, San, color = 'b', linewidth = 1.5)
plt.xlabel(r'$T$ (s)')
plt.ylabel(r'$S_a$ (g)')
plt.axis([Tn[0], Tn[-1], 0.0, 0.8])
plt.grid(True)
plt.show
plt.savefig(f'Espectro_NSR10_Suelo_{TS}.png', dpi = 300, bbox_inches = 'tight')

NGDL = nPy
ni = Nmodes if ni == 0 else ni

Dx, Δx, Vx, df2 = AnalisisModal(Aa, Av, I, TS, Ms, modo, Tmodes, NGDL, ni, ux, FA)

print('Análsis Modal Espectral')
df2 = df2.astype({'Nivel': int})
display(df2.round(2))

df3 = pd.DataFrame(columns = ['Nivel', 'Vx (kN)', 'Dx (cm)', 'Δx (%)'])
for i in range(nPy):
    rΔx = Δx[i]/dHy # relación de deriva de entrepiso en x
    df3 = df3._append({'Nivel': i+1, 'Vx (kN)': Vx[i]*1e-3,
                      'Dx (cm)': Dx[i]*1e2,
                      'Δx (%)': rΔx*1e2}, ignore_index = True)
df3 = df3.astype({'Nivel': int})
display(df3.round(2))

# Graficar la relación de deriva y cortante basal
vecx = np.array(df3.loc[:,'Δx (%)'])
vecV = np.array(df3.loc[:,'Vx (kN)'])
vecVs = np.append(np.repeat(vecV, 2), 0)
vpis = np.repeat(np.arange(nPy+1), 2)[1:]
Vst = vecV.max()
Dermax = vecx.max()

lim = 1.1*vecx.max()
lim2 = 1.1*vecV.max()

plt.figure(figsize = (4.0, 8.0))
plt.plot(np.insert(vecx,0,0), np.arange(nPy+1), 'o-', color = 'crimson', label = 'en x', lw = 1.6, ms = 6)
plt.legend()
plt.xlabel('Relación de deriva (%)')
plt.ylabel('Nivel')
plt.axis([-0.1, lim+0.1, -0.1, nPy+0.1])
plt.grid(True)
plt.show()
plt.savefig('Relacion_de_Deriva.png', dpi = 300, bbox_inches = 'tight')

plt.figure(figsize = (4.0, 8.0))
plt.plot(vecVs, vpis, '-', color = 'crimson', label = 'en x', lw = 1.6, ms = 6)
plt.legend()
plt.xlabel('Cortante (kN)')
plt.ylabel('Nivel')
plt.axis([-0.1, lim2+0.1, -0.1, nPy+0.1])
plt.grid(True)
plt.show()
plt.savefig('Cortante_Basal.png', dpi = 300, bbox_inches = 'tight')

# Cálculo del corte mínimo
Ct, alpha = 0.047, 0.9 # Tabla A.4.2-1
Hedf = nPy*dHy # altura de la edificación (m)
Ta = Ct*Hedf**alpha # Periodo aproximado (s)
Saa = SaNSR10(Ta, Aa, Av, I, TS)
Vs = Mux*Saa*g*1e-3
FaVs = 0.8*Vs/Vst

print('---------------------------------------')
print('Fuerza Horizontal Equivalente')
print('---------------------------------------')
print(f'T_a = {round(Ta, 2)} s')
print(f'V_s = {round(Vs, 2)} kN')
print('---------------------------------------')
print('Superposición modal')
print('---------------------------------------')
print(f'Cantidad de modos requeridos: {nmreq}')
print(f'Deriva_máxima = {round(Dermax, 2)} %')
print(f'Vst = {round(Vst, 2)}')
print('Mantener FA = 1.00' if FaVs <= 1.0 else f'Modificar FA = {round(FaVs, 2)}')

modeNo = 1  # especificar el modo que se quiere animar
fmt_defo = {'color': 'blue', 'linestyle': 'solid', 'linewidth': 3.0,
            'marker': '', 'markersize': 6}
deltaa = 3
xlima = [-deltaa, nVx*LVx+deltaa]
ylima = [-deltaa, nPy*dHy+deltaa]
anim = opsv.anim_mode(modeNo, fmt_defo=fmt_defo, fmt_undefo='g--', xlim=xlima, ylim=ylima,
                      fig_lbrt=(0.04, 0.04, 0.96, 0.96), fig_wi_he=(30., 22.))
plt.title(f'Mode {modeNo}, T_{modeNo}: {Tmodes[1]:.3f} s')
plt.show()

# Guardar como mp4 usando ffmpeg writer (instalar: conda install -c conda-forge ffmpeg)
import matplotlib.animation as animation
writer = animation.FFMpegWriter(fps=15, bitrate=1800)
anim.save(f'Modo_{modeNo}.mp4', writer=writer)
