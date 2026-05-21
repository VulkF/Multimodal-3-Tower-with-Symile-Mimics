# Multimodal Clinical Diagnosis v4 (CXR, ECG & Labs)

## 📌 Descripción del Proyecto
Este repositorio contiene la implementación de una **Arquitectura Multimodal de "Tres Torres" (Late Fusion)** diseñada para el diagnóstico automatizado en entornos de cuidados intensivos. Inspirado en modelos como NYUTron, el sistema cruza información de tres modalidades clínicas diferentes para emitir diagnósticos robustos (ej. Cardiomegalia, Edema, Derrame Pleural, Atelectasia).

## 🧠 Arquitectura del Modelo
El modelo utiliza un enfoque de fusión tardía complementado con Atención Cruzada (*Cross-Modal Attention*), emulando el razonamiento clínico humano:
1. **Torre Visual (CXR):** `EfficientNet-B3` (preentrenada en ImageNet) + *Multi-Window Self-Attention* para extraer patrones espaciales de radiografías (320x320).
2. **Torre de Señales (ECG):** `1D CNN` con bloques *Squeeze-and-Excitation (SE)* para calibrar la importancia de las 12 derivaciones cardíacas.
3. **Torre Tabular (Labs):** `Perceptrón Multicapa (MLP)` con conexiones residuales para estabilizar la alta varianza de 50 percentiles de laboratorio.
4. **Fusión:** La radiografía actúa como *Query* y el contexto fisiológico (ECG + Labs) actúa como *Key/Value* en una capa de atención cruzada, generando un vector de predicción final.

## 📊 Métricas Principales
El modelo fue evaluado priorizando la métrica **ROC-AUC** debido al desbalance natural de las patologías médicas, logrando superar a las líneas base de modalidades individuales y demostrando alta tolerancia a fallos ante la ausencia de ciertas pruebas clínicas.

## 🛠️ Tecnologías Utilizadas
* Python 3.x
* PyTorch
* PyTorch Image Models (timm)
* Scikit-Learn
* Matplotlib & Seaborn (Dashboard de métricas)

* ## 📂 Acceso al Dataset (Symile-MIMIC / MIMIC-IV)
Debido a regulaciones de privacidad médica (HIPAA) y al Acuerdo de Uso de Datos (DUA), **los datos clínicos no están incluidos en este repositorio**.

Para ejecutar este código, debes descargar el dataset oficial siguiendo estos pasos:
1. Regístrate en [PhysioNet](https://physionet.org/).
2. Completa el curso de entrenamiento ético requerido (CITI Training).
3. Firma el acuerdo de uso de datos para acceder a **MIMIC-IV** y **MIMIC-CXR**.
4. (Opcional/Específico) Descarga la versión pre-procesada de **Symile-MIMIC** desde su [repositorio oficial en PhysioNet](https://physionet.org/content/symile-mimic/).

**Estructura esperada de los datos:**
El código asume que los datos preprocesados se encuentran en formato `.npy` con las siguientes dimensiones:
* `cxr_<split>.npy` -> `(n, 3, 320, 320)`
* `ecg_<split>.npy` -> `(n, 1, 5000, 12)`
* `labs_percentiles_<split>.npy` -> `(n, 50)`
