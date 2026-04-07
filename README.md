    Tensor en C++
Implementación de una estructura de tensor utilizando memoria contigua en C++

Descripción:
En el proyecto se desarrolla una clase Tensor capaz de manejar datos multidimensionales mediante un arreglo lineal en memoria. Tiene operaciones básicas de manipulación de dimensiones y generación de datos.

Funcionalidades: 
reshape, view (sin copia de memoria), unsqueeze, concat, arange, generación de valores aleatorios

Tecnologías: C++, Git y Github

Compilación y ejecución:

Compilación: g++ main.cpp -o main

Ejecución: Para Windows-> main.exe, para Linux/macOS->
./main

Comandos principales:
Tensor({dimensiones})-> crea un tensor indicando el tamaño en cada dimensión.
t(i, j, ...) -> accede o modifica un elemento del tensor mediante índices.
t.print() -> muestra el contenido del tensor en consola

El programa controla errores como: índices fuera de rango, dimensiones inválidas. 

Estudiantes: Alvarez Lovera Sandra Sofia, Davila Bazan Santiago Sebastian

