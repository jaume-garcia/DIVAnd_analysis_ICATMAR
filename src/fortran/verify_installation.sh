#!/bin/bash

# Script para verificar la instalación y compilación

echo "======================================================================"
echo "Script de Verificación - HF Radar Processor Fortran"
echo "======================================================================"
echo ""

# Función para imprimir con color
print_success() {
    echo -e "\033[0;32m✓ $1\033[0m"
}

print_error() {
    echo -e "\033[0;31m✗ $1\033[0m"
}

print_warning() {
    echo -e "\033[0;33m⚠ $1\033[0m"
}

print_info() {
    echo -e "\033[0;34mℹ $1\033[0m"
}

# Verificar compilador Fortran
echo "1. Verificando compilador Fortran..."
if command -v gfortran &> /dev/null; then
    VERSION=$(gfortran --version | head -n1)
    print_success "gfortran encontrado: $VERSION"
else
    print_error "gfortran no encontrado"
    echo "   Instala con: sudo apt-get install gfortran"
    exit 1
fi
echo ""

# Verificar NetCDF
echo "2. Verificando NetCDF..."
if command -v nf-config &> /dev/null; then
    VERSION=$(nf-config --version)
    print_success "NetCDF-Fortran encontrado: $VERSION"
    print_info "Include path: $(nf-config --includedir)"
    print_info "Library path: $(nf-config --libdir)"
else
    print_warning "nf-config no encontrado"
    echo "   Verifica la instalación de NetCDF-Fortran"
    echo "   Ubuntu: sudo apt-get install libnetcdff-dev"
    echo "   macOS: brew install netcdf-fortran"
fi
echo ""

# Verificar archivos fuente
echo "3. Verificando archivos fuente..."
REQUIRED_FILES=("hfradar_module.f90" "hfradar_main.f90" "Makefile")
ALL_PRESENT=true

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "$file presente"
    else
        print_error "$file NO encontrado"
        ALL_PRESENT=false
    fi
done

if [ "$ALL_PRESENT" = false ]; then
    echo ""
    print_error "Faltan archivos necesarios"
    exit 1
fi
echo ""

# Verificar sintaxis
echo "4. Verificando sintaxis de archivos Fortran..."
if gfortran -fsyntax-only hfradar_module.f90 2>/dev/null; then
    print_success "hfradar_module.f90 - sintaxis correcta"
else
    print_error "hfradar_module.f90 - errores de sintaxis"
    gfortran -fsyntax-only hfradar_module.f90
    exit 1
fi

if gfortran -fsyntax-only hfradar_main.f90 2>/dev/null; then
    print_success "hfradar_main.f90 - sintaxis correcta"
else
    print_error "hfradar_main.f90 - errores de sintaxis"
    gfortran -fsyntax-only hfradar_main.f90
    exit 1
fi
echo ""

# Intentar compilar
echo "5. Intentando compilación..."
if make clean > /dev/null 2>&1 && make > /dev/null 2>&1; then
    print_success "Compilación exitosa"
    
    # Verificar que el ejecutable existe
    if [ -f "hfradar_program" ]; then
        print_success "Ejecutable creado: hfradar_program"
        
        # Mostrar tamaño del ejecutable
        SIZE=$(ls -lh hfradar_program | awk '{print $5}')
        print_info "Tamaño del ejecutable: $SIZE"
    else
        print_error "Ejecutable no encontrado después de compilación"
        exit 1
    fi
else
    print_error "Error en la compilación"
    echo ""
    echo "Intentando compilación con salida detallada..."
    make clean
    make
    exit 1
fi
echo ""

# Verificar estructura de directorios para datos
echo "6. Verificando estructura de directorios..."
print_info "El programa espera encontrar datos en:"
print_info "  - Entrada: /home/jgarcia/Projects/mar_catala/radial_reconstruction/data/..."
print_info "  - Salida:  /home/jgarcia/Projects/mar_catala/radial_reconstruction/data/..."
print_info "  - Grilla:  /home/jgarcia/Projects/mar_catala/radial_reconstruction/data/..."
echo ""
print_warning "Asegúrate de que estas rutas existan o modifica hfradar_main.f90"
echo ""

# Resumen
echo "======================================================================"
echo "Resumen de Verificación"
echo "======================================================================"
print_success "Compilador Fortran: OK"
if command -v nf-config &> /dev/null; then
    print_success "NetCDF: OK"
else
    print_warning "NetCDF: Verifica instalación"
fi
print_success "Archivos fuente: OK"
print_success "Sintaxis: OK"
print_success "Compilación: OK"
echo ""
print_info "Para ejecutar: ./hfradar_program"
print_info "Para limpiar: make clean"
print_info "Para ayuda: make help"
echo ""

# Crear un archivo de ejemplo de entrada si no existe
if [ ! -f "test_input.txt" ]; then
    echo "Creando archivo de prueba test_input.txt..."
    cat > test_input.txt << EOF
2.500 41.500 10.5 -5.2 45.0 11.7 45.0 1
2.505 41.505 12.1 -6.8 47.2 13.9 47.2 1
2.510 41.510 8.3 -4.1 43.5 9.2 43.5 2
2.515 41.515 9.7 -5.9 46.8 11.4 46.8 2
2.520 41.520 11.2 -7.1 48.1 13.2 48.1 1
EOF
    print_success "Archivo de prueba creado: test_input.txt"
    echo ""
fi

echo "======================================================================"
print_success "Verificación completada exitosamente"
echo "======================================================================"
