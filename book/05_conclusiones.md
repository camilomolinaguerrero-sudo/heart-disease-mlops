# Conclusiones

- **Fuga de datos.** Una variable que contiene la respuesta lleva el AUC de prueba a 1,000 con o sin
  `Pipeline`: ninguna técnica de validación corrige una variable contaminada, solo la revisión de las
  variables. El escalado global es una fuga real, pero aquí casi inocua (MinMax solo filtra mínimos y máximos;
  AUC 0,928 en ambos casos).
- **Modelo.** Con 184 pacientes de prueba, los cinco modelos comparados difieren en 0,02 de AUC, así que
  ninguno se impone con claridad. Se eligió RandomForest solo por AUC de validación cruzada (0,9315). En
  prueba: AUC 0,928, accuracy 0,880, precision 0,870, recall 0,922, F1 0,895. La prueba de permutación
  (100 permutaciones) da AUC 0,500 ± 0,029 con la respuesta barajada (p = 0,0099, el mínimo posible con 100).
- **Despliegue.** El mismo modelo responde igual en el entorno local, en Docker y en Kubernetes.
- **Limitaciones.** El conjunto es pequeño y proviene de un espejo público; el kind de una sola máquina no
  prueba escalado real; el umbral de alerta de deriva por defecto es poco sensible.
