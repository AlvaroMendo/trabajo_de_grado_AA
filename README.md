
# EFECTO DE LA VARIACIÓN DEL MÓDULO DE ELASTICIDAD EN LA DERIVA EN PÓRTICOS DE CONCRETO REFORZADO DE MEDIANA ALTURA, UBICADAS EN LA CIUDAD DE MONTERÍA

**Alejandra Sepúlveda Arteaga**  
**Álvaro Andrés Mendoza Castellanos**  

> Trabajo de grado presentado para optar al título de Ingeniero Civil  

**Asesor**  
John Esteban Ardila González, Magíster en Ingeniería Civil  

**Universidad Pontificia Bolivariana**  
Escuela de Ingenierías y Arquitectura  
Programa de Ingeniería Civil  
Montería, 2025  

---

## 📌 Descripción

Este repositorio contiene el modelo numérico desarrollado en **OpenSeesPy** para el análisis modal espectral de un pórtico plano de concreto reforzado de cinco niveles. El objetivo es estudiar cómo la variación del módulo de elasticidad del concreto influye en la **deriva de entrepiso**.

Se implementa un enfoque académico que incluye la creación de nodos y elementos estructurales, definición de materiales, masas, diafragmas rígidos y cálculo de los primeros modos de vibración.

---

## 🧠 Contenido del código

- **Lenguaje:** Python 3  
- **Framework de análisis:** [OpenSeesPy](https://openseespydoc.readthedocs.io/en/latest/)
- **Visualización:** `opsvis`, `matplotlib`
- **Resultados:** Tabla de periodos, frecuencias y contribución modal

---

## 🔧 Parámetros principales

- Piso: `5 niveles`  
- Vano: `3 vanos de 5.0 m`  
- Altura de piso: `3.0 m`  
- Sección columna: `30x30 cm`  
- Sección viga: `30x50 cm`  
- Resistencia del concreto: `f'c = 21 MPa`  
- Relación módulo de elasticidad: `Ec = 4700 √f'c`  

---

## 📊 Resultados generados

- Tablas con frecuencias naturales y periodos  
- Modos de vibración visualizados en imágenes `.png`  
- Contribución de masa modal acumulada  
- Representación del modelo estructural (nodos y elementos)

---

## ▶️ Ejecución

Para correr el modelo:

```bash
python nombre_del_archivo.py
```

> Asegúrate de tener instaladas las librerías necesarias:

```bash
pip install openseespy matplotlib opsvis pandas
```

---

## 📁 Salidas gráficas

- `NodElemP2D.png`: Geometría del pórtico  
- `FormaModal_1.png`, `FormaModal_2.png`, `FormaModal_3.png`: Modos de vibración  

---

## 📚 Citación sugerida

Si deseas referenciar este trabajo:

> Sepúlveda A., Mendoza Á. (2025). *Efecto de la variación del módulo de elasticidad en la deriva en pórticos de concreto reforzado de mediana altura, ubicadas en la ciudad de Montería*. Universidad Pontificia Bolivariana, Montería.

---

## 🤝 Contacto

Para más información o colaboración académica, puedes escribirnos:

- alejandra.sep@ejemplo.edu.co  
- alvaro.mendoza@ejemplo.edu.co  

---
