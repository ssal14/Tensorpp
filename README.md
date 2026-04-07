    Tensor en C++
Implementación de una estructura de tensor utilizando memoria contigua en C++

Descripción:
En el proyecto se desarrolla una clase Tensor capaz de manejar datos multidimensionales mediante un arreglo lineal en memoria. Tiene operaciones básicas de manipulación de dimensiones y generación de datos.

Funcionalidades: 
construcción con shape y values, generación de tensores (zeros, ones, random, arange), cambio de forma con view, inserción de dimensión con unsqueeze, concatenación con concat, operaciones (suma, resta, multiplicación elemento a elemento, multiplicación por escalar), funciones amigas (dot, matmul), transformaciones (ReLu, Sigmoid)

Tecnologías: C++, Git y Github

Compilación y ejecución:

Compilación: g++ main.cpp -o main

Comandos principales:
Tensor({dimensiones})-> crea un tensor indicando el tamaño en cada dimensión.
t(i, j, ...) -> accede o modifica un elemento del tensor mediante índices.
t.print() -> muestra el contenido del tensor en consola

El programa controla errores como: índices fuera de rango, dimensiones inválidas. 

Estudiantes: Alvarez Lovera Sandra Sofia, Davila Bazan Santiago Sebastian

